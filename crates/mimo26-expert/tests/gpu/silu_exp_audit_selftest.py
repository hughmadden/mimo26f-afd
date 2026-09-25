#!/usr/bin/env python3
"""Synthetic receipt integrity controls, never native numerical evidence."""
import copy
import math
import struct
import silu_exp_decimal as audit


def fixture():
    counts = dict(patterns=2**32, finite=4278190080, nan=16777214, infinite=2,
                  near_midpoints=1)
    cpu = dict(counts, reference_fnv64='1234')
    summary = dict(counts, corrected_reference_fnv64='1234', special_bad=0,
                   reference_corrections=0, zero_one_infinity_exp_mismatches=0,
                   activation_exp2_set_nonmembers_u1=0)
    examples, worst = [], []
    for index,key in enumerate(audit.DOMAINS):
        b = 0x3f800000 | ((index % 2) << 31)
        x = audit.f32(b)
        if index < 2:
            want = audit.checked_exp(x)[0]
        else:
            e = audit.f32(audit.checked_exp(-abs(x))[0])
            rn = lambda v: audit.f32(audit.bits32(v))
            want = audit.bits32(rn((x if x >= 0 else rn(x*e))/rn(1.+e)))
        got = want+1
        summary[key] = dict(mismatch=1,max_ulp=1,input_bits=b,got_bits=got,reference_bits=want)
        row = dict(kind='exp' if index < 2 else 'activation-u1',input_bits=f'{b:08x}',
                   gpu_bits=f'{got:08x}',reference_bits=f'{want:08x}')
        examples.append(row)
        worst.append(dict(row,kind=row['kind']+'-worst'))
    y,mid = math.exp(2**-24),1.+2**-24
    bits64 = lambda v: f"{struct.unpack('<Q',struct.pack('<d',v))[0]:016x}"
    reference = f'{audit.bits32(y):08x}'
    near = [dict(input_bits='33800000',reference_bits=reference,double_bits=bits64(y),
                 midpoint_bits=bits64(mid),gap_double_ulps=str(abs(y-mid)/(math.nextafter(y,math.inf)-y)),
                 gpu_exp_bits=reference,corrected_reference_bits=reference)]
    return summary,cpu,examples+worst,near


def main():
    base = fixture()
    audit.integrity(*base)
    controls = []
    def control(name, change):
        args = copy.deepcopy(base)
        change(*args)
        try:
            audit.integrity(*args)
        except (ValueError, KeyError):
            controls.append(name)
        else:
            raise AssertionError('unpowered receipt mutation: '+name)
    for source in (0,1):
        control(f'partial-coverage-{source}',lambda s,c,e,m,i=source: (s,c)[i].update(patterns=2**32-1))
        control(f'class-count-swap-{source}',lambda s,c,e,m,i=source: (s,c)[i].update(finite=4278190081,nan=16777213))
    control('boolean-count',lambda s,c,e,m: s.update(infinite=True))
    control('checksum-mismatch',lambda s,c,e,m: c.update(reference_fnv64='1235'))
    def invalid_hash(s,c,e,m):
        s['corrected_reference_fnv64']=c['reference_fnv64']='not-a-hash'
    control('invalid-checksum',invalid_hash)
    for key in ('special_bad','reference_corrections','zero_one_infinity_exp_mismatches','activation_exp2_set_nonmembers_u1'):
        control('negative-'+key,lambda s,c,e,m,k=key: s.update({k:-1}))
    control('special-class-failure',lambda s,c,e,m: s.update(special_bad=1))
    control('missing-midpoint',lambda s,c,e,m: m.clear())
    control('duplicate-midpoint',lambda s,c,e,m: (m.append(dict(m[0])),s.update(near_midpoints=2)))
    control('false-midpoint',lambda s,c,e,m: m[0].update(midpoint_bits='3ff0000000000000'))
    control('nan-midpoint-gap',lambda s,c,e,m: m[0].update(gap_double_ulps='nan'))
    control('false-midpoint-gap',lambda s,c,e,m: m[0].update(gap_double_ulps='0'))
    control('correction-count',lambda s,c,e,m: s.update(reference_corrections=1))
    control('missing-example',lambda s,c,e,m: e.pop(0))
    control('missing-worst',lambda s,c,e,m: e.pop())
    control('duplicate-example',lambda s,c,e,m: e.insert(0,dict(e[0])))
    control('unknown-kind',lambda s,c,e,m: e[0].update(kind='activation-other'))
    control('not-a-mismatch',lambda s,c,e,m: e[0].update(gpu_bits=e[0]['reference_bits']))
    control('nonfinite-input',lambda s,c,e,m: e[0].update(input_bits='7f800000'))
    control('nan-exp-output',lambda s,c,e,m: e[0].update(gpu_bits='7fc00000'))
    control('infinite-activation',lambda s,c,e,m: e[2].update(gpu_bits='7f800000'))
    control('negative-mismatches',lambda s,c,e,m: s['exp_positive'].update(mismatch=-1))
    control('too-many-mismatches',lambda s,c,e,m: s['exp_positive'].update(mismatch=2139095041))
    control('missing-domain',lambda s,c,e,m: s.pop('exp_negative'))
    control('zero-max-with-mismatches',lambda s,c,e,m: s['exp_positive'].update(max_ulp=0))
    control('wrong-ulp-distance',lambda s,c,e,m: s['exp_positive'].update(max_ulp=2))
    control('altered-worst',lambda s,c,e,m: e[-1].update(gpu_bits='80000000'))
    control('endpoint-overcount',lambda s,c,e,m: s.update(zero_one_infinity_exp_mismatches=3))
    control('nonmember-overcount',lambda s,c,e,m: s.update(activation_exp2_set_nonmembers_u1=3))
    def bad_domain(s,c,e,m):
        s['exp_positive']['input_bits'] |= 0x80000000
        e[4]['input_bits']=f"{s['exp_positive']['input_bits']:08x}"
    control('wrong-worst-domain',bad_domain)
    zero = copy.deepcopy(base)
    zero[2][:] = zero[2][4:]
    for i,key in enumerate(audit.DOMAINS):
        zero[0][key] = dict(mismatch=0,max_ulp=0,input_bits=0,got_bits=0,reference_bits=0)
        zero[2][i].update(input_bits='00000000',gpu_bits='00000000',reference_bits='00000000')
    audit.integrity(*zero)
    print(f'HOST PASS receipt integrity: 2 synthetic positives, {len(controls)} powered corruptions; no GPU or native-evidence claim')
    print('CONTROLS '+', '.join(controls))


if __name__ == '__main__':
    main()
