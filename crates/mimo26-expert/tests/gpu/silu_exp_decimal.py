#!/usr/bin/env python3
"""Independent Decimal rounding audit; never imports the CUDA implementation."""
import argparse
import csv
import json
import math
import struct
from decimal import Decimal, localcontext
from pathlib import Path


def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bits32(value):
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        return 0x7f800000 if value > 0 else 0xff800000


def d32(bits):
    return Decimal.from_float(f32(bits))


def nearest(value):
    # All exp outputs are positive. Binary32 overflow uses a virtual2^128
    # successor, not distance to Decimal Infinity.
    with localcontext() as ctx:
        ctx.prec = 220
        overflow = Decimal(2)**128 - Decimal(2)**103
        if value >= overflow:
            return 0x7f800000
        if value <= d32(1)/2:
            return 0
        base = min(bits32(float(value)), 0x7f7fffff)
        return min(range(max(0, base-2), min(0x7f7fffff, base+2)+1),
                   key=lambda b: (abs(d32(b)-value), b & 1))


def checked_exp(x):
    values = []
    with localcontext() as ctx:
        for precision in (110, 180):
            ctx.prec = precision
            y = Decimal.from_float(x).exp()  # documented correctly rounded RNE
            ctx.prec = 220
            radius = Decimal(10)**(y.adjusted()-precision+1)/2
            low, high = nearest(y-radius), nearest(y+radius)
            if low != high:
                raise ValueError('Decimal rounding interval straddles binary32 midpoint')
            values.append((low, str(y), str(radius)))
    if values[0][0] != values[1][0]:
        raise ValueError('Decimal precision disagreement')
    return values[-1]


def selftest():
    with localcontext() as ctx:
        ctx.prec = 220
        for low in (0, 1, 2, 3, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x7f7ffffe):
            midpoint = (d32(low)+d32(low+1))/2
            delta = (d32(low+1)-d32(low))/100000
            assert nearest(midpoint) == (low if low % 2 == 0 else low+1)
            assert nearest(midpoint-delta) == low
            assert nearest(midpoint+delta) == low+1
        boundary = Decimal(2)**128-Decimal(2)**103
        assert nearest(boundary-1) == 0x7f7fffff and nearest(boundary) == 0x7f800000
    assert checked_exp(-0.1987401247024536)[0] == 1062329339
    assert checked_exp(-103.)[0] == 1
    assert checked_exp(-104.)[0] == 0
    # Check the generous analytic shortcuts independently at their extrema.
    assert checked_exp(90.)[0] == 0x7f800000
    assert checked_exp(-105.)[0] == 0
    assert checked_exp(2**-26)[0] == checked_exp(-2**-26)[0] == 0x3f800000
    print('HOST PASS Decimal110/180: midpoint sides/ties-even, gradual underflow, overflow and analytic-region boundaries')


def midpoint_metadata(row):
    b, reference = int(row['input_bits'],16), int(row['reference_bits'],16)
    y = struct.unpack('<d', struct.pack('<Q', int(row['double_bits'],16)))[0]
    mid = struct.unpack('<d', struct.pack('<Q', int(row['midpoint_bits'],16)))[0]
    if not math.isfinite(f32(b)) or not math.isfinite(y) or y <= 0 or bits32(y) != reference:
        raise ValueError('bad midpoint input/reference')
    center = 2.**128 if reference == 0x7f800000 else f32(reference)
    lower = f32(reference-1) if reference else 0.
    upper = 2.**128 if reference >= 0x7f7fffff else f32(reference+1)
    ml, mu = (lower+center)/2, (center+upper)/2
    expected_mid = ml if reference == 0x7f800000 else mu if reference == 0 else ml if abs(y-ml) < abs(y-mu) else mu
    ulp = math.nextafter(y, math.inf)-y
    gap = abs(y-mid)/ulp
    if mid != expected_mid or not 0 <= gap <= 256 or float(row['gap_double_ulps']) != gap:
        raise ValueError('invalid midpoint geometry/distance')
    return b, reference, y, ulp


def audit(path, output):
    rows = []
    seen = set()
    for row in csv.DictReader(path.open()):
        b, reference, double, double_ulp = midpoint_metadata(row)
        if b in seen:
            raise ValueError('duplicate midpoint input')
        seen.add(b)
        correct, high, radius = checked_exp(f32(b))
        with localcontext() as ctx:
            ctx.prec = 220
            if abs(Decimal.from_float(double)-Decimal(high)) > 256*Decimal.from_float(double_ulp)+Decimal(radius):
                raise ValueError('audited binary64 result exceeds the declared error guard')
        if 'corrected_reference_bits' in row and int(row['corrected_reference_bits'],16) != correct:
            raise ValueError('GPU comparison used a noncanonical near-midpoint reference')
        rows.append(dict(input_bits=f'{b:08x}', reference_bits=f'{reference:08x}',
                         correct_bits=f'{correct:08x}', differs=correct != reference,
                         decimal180=high, decimal_error_radius=radius,
                         gap_double_ulps=float(row['gap_double_ulps'])))
    if not rows:
        raise ValueError('empty near-midpoint audit')
    output.mkdir(parents=True, exist_ok=True)
    (output/'reference-map.tsv').write_text('# input_bits correctly_rounded_exp_bits; all audited near-midpoint cases\n'+
        ''.join(f"{r['input_bits']} {r['correct_bits']}\n" for r in rows))
    result = dict(scope='Decimal interval audit of all <=256 binary64-ULP midpoint candidates; not an unproved global libm error guarantee',
                  audited=len(rows), corrections=sum(r['differs'] for r in rows), rows=rows)
    (output/'decimal.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"DECIMAL PASS audited={len(rows)} binary64-then-round corrections={result['corrections']}")


DOMAINS = ('exp_positive', 'exp_negative', 'activation_positive_u1', 'activation_negative_u1')


def require(value, message):
    if not value:
        raise ValueError(message)


def uint(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def distance(a, b):
    def ordered(v):
        return (~v & 0xffffffff) if v >> 31 else v ^ 0x80000000
    return abs(ordered(a)-ordered(b))


def integrity(summary, cpu, examples, midpoints):
    coverage = dict(patterns=2**32, finite=4278190080, nan=16777214, infinite=2)
    for record in (summary, cpu):
        for key, expected in coverage.items():
            require(type(record[key]) is int and record[key] == expected, 'incomplete/class-shifted enumeration')
    require(summary['corrected_reference_fnv64'] == cpu['reference_fnv64'], 'cross-host reference checksum differs')
    fingerprint = cpu['reference_fnv64']
    require(isinstance(fingerprint,str) and 1 <= len(fingerprint) <= 16 and all(c in '0123456789abcdef' for c in fingerprint), 'malformed reference checksum')
    for key in ('special_bad', 'near_midpoints', 'reference_corrections', 'zero_one_infinity_exp_mismatches', 'activation_exp2_set_nonmembers_u1'):
        require(uint(summary[key],4278190080), 'invalid diagnostic count')
    require(summary['special_bad'] == 0, 'nonfinite exp class failure')
    require(uint(cpu['near_midpoints'],4278190080) and cpu['near_midpoints'] > 0, 'invalid CPU midpoint count')
    require(summary['near_midpoints'] == len(midpoints) > 0, 'missing/truncated midpoint records')
    inputs = [int(r['input_bits'],16) for r in midpoints]
    require(len(set(inputs)) == len(inputs), 'duplicate midpoint records')
    for row in midpoints:
        midpoint_metadata(row)
        require(uint(int(row['gpu_exp_bits'],16),0x7f800000), 'invalid exp witness bits')
    corrections = sum(int(r['reference_bits'],16) != int(r['corrected_reference_bits'],16) for r in midpoints)
    require(summary['reference_corrections'] == corrections, 'reference correction count mismatch')
    # This revision compares a raw CPU checksum to a corrected GPU checksum.
    require(corrections == 0, 'nonzero corrections need a separately corrected CPU fingerprint')
    regular = {key: [] for key in DOMAINS}
    worst = []
    seen = set()
    for row in examples:
        kind = row['kind']
        require(kind in ('exp','activation-u1','exp-worst','activation-u1-worst'), 'unknown example kind')
        b, got, want = (int(row[k],16) for k in ('input_bits','gpu_bits','reference_bits'))
        require(all(uint(v,0xffffffff) for v in (b,got,want)), 'invalid example word')
        if kind.endswith('-worst'):
            worst.append(row)
            continue
        require(math.isfinite(f32(b)) and got != want, 'not a finite counterexample')
        identity = (kind,b)
        require(identity not in seen, 'duplicate counterexample')
        seen.add(identity)
        index = (2 if kind == 'activation-u1' else 0)+(b >> 31)
        require(all(math.isfinite(f32(v)) for v in (got,want)) if index >= 2 else all(v <= 0x7f800000 for v in (got,want)), 'invalid numerical output class')
        regular[DOMAINS[index]].append(row)
    require(len(worst) == 4, 'missing/extra worst-case records')
    for index,key in enumerate(DOMAINS):
        stat = summary[key]
        require(uint(stat['mismatch'],2139095040) and uint(stat['max_ulp'],0xffffffff), 'invalid per-sign statistics')
        require((stat['mismatch'] == 0) == (stat['max_ulp'] == 0), 'mismatch/max inconsistency')
        require(all(uint(stat[k],0xffffffff) for k in ('input_bits','got_bits','reference_bits')), 'invalid worst-case words')
        row = worst[index]
        require(row['kind'] == ('exp-worst' if index < 2 else 'activation-u1-worst'), 'wrong worst-case ordering')
        for csv_key,stat_key in (('input_bits','input_bits'),('gpu_bits','got_bits'),('reference_bits','reference_bits')):
            require(int(row[csv_key],16) == stat[stat_key], 'worst-case summary/CSV mismatch')
        if stat['mismatch']:
            require(all(math.isfinite(f32(stat[k])) for k in ('got_bits','reference_bits')) if index >= 2 else all(stat[k] <= 0x7f800000 for k in ('got_bits','reference_bits')), 'invalid worst numerical output class')
            require(math.isfinite(f32(stat['input_bits'])) and stat['input_bits'] >> 31 == index % 2, 'wrong worst-case domain')
            require(distance(stat['got_bits'],stat['reference_bits']) == stat['max_ulp'], 'wrong worst-case ULP distance')
        else:
            require(all(stat[k] == 0 for k in ('input_bits','got_bits','reference_bits')), 'bad zero-mismatch placeholder')
        require(len(regular[key]) == min(24,stat['mismatch']), 'missing/extra retained counterexamples')
        for sample in regular[key]:
            require(distance(int(sample['gpu_bits'],16),int(sample['reference_bits'],16)) <= stat['max_ulp'], 'counterexample exceeds reported maximum')
    exp_bad = sum(summary[k]['mismatch'] for k in DOMAINS[:2])
    activation_bad = sum(summary[k]['mismatch'] for k in DOMAINS[2:])
    require(summary['zero_one_infinity_exp_mismatches'] <= exp_bad, 'endpoint count exceeds exp mismatches')
    require(summary['activation_exp2_set_nonmembers_u1'] <= activation_bad, 'nonmember count exceeds activation mismatches')


def gpu_audit(root, cpu_summary, output=None):
    summary = json.loads((root/'gpu-summary.json').read_text())
    cpu = json.loads(cpu_summary.read_text())
    examples = list(csv.DictReader((root/'gpu-examples.csv').open()))
    midpoints = list(csv.DictReader((root/'gpu-midpoints.csv').open()))
    integrity(summary,cpu,examples,midpoints)
    checked = []
    def rn(value):
        return f32(bits32(value))
    worst_index = 0
    for row in examples:
        if row['kind'].endswith('-worst'):
            stat = summary[DOMAINS[worst_index]]
            worst_index += 1
            if stat['mismatch'] == 0:
                continue  # Native zero-mismatch placeholder, not an exp(0) claim.
        b = int(row['input_bits'],16)
        x = f32(b)
        if row['kind'].startswith('exp'):
            correct = checked_exp(x)[0] if -105 <= x <= 90 else (0 if x < 0 else 0x7f800000)
        else:
            e = f32(checked_exp(-abs(x))[0]) if abs(x) <= 105 else 0.
            numerator = x if x >= 0 else rn(x*e)
            correct = bits32(rn(numerator/rn(1.+e)))
        if correct != int(row['reference_bits'],16):
            raise ValueError('Decimal counterexample reference mismatch')
        checked.append(dict(**row, decimal_reference_bits=f'{correct:08x}',
                            bitwise_match=correct == int(row['gpu_bits'],16)))
    result = dict(scope='independent Decimal audit of retained GPU counterexamples; global counts from native exhaustive scan',
                  checked=len(checked), cross_host_reference_fnv64_match=True, rows=checked)
    output = output or root
    output.mkdir(parents=True, exist_ok=True)
    (output/'gpu-example-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"GPU AUDIT COMPLETE patterns={summary['patterns']} examples={len(checked)} exp_negative_max_ulp={summary['exp_negative']['max_ulp']} bitwise_match={summary['exp_negative']['mismatch']==0}; no lattice change")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu-root', type=Path)
    p.add_argument('--cpu-summary', type=Path)
    p.add_argument('--gpu-output', type=Path)
    a = p.parse_args()
    selftest()
    if not a.selftest:
        if not a.input or not a.output:
            p.error('--input and --output required')
        audit(a.input, a.output)
        if a.gpu_root:
            if not a.cpu_summary:
                p.error('--cpu-summary required for GPU audit')
            gpu_audit(a.gpu_root, a.cpu_summary, a.gpu_output)
