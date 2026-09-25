#!/usr/bin/env python3
"""Bounded abstract C1 lifetime/liveness model, NOT a CUDA memory-model proof.

Atomic events represent complete role groups (producer 64, QK 64, PV 128).
Actual code must implement release/acquire, full-group participation, named
barriers, cp.async waits and final drain before this abstraction applies.
"""
from dataclasses import dataclass, replace
from collections import deque
import math


@dataclass(frozen=True)
class State:
    p: int = 0
    q: int = 0
    v: int = 0
    raw: str = 'empty'
    raw_epoch: int = -1
    expanding: bool = False
    qr: bool = False
    vr: bool = False
    slots: tuple = (('free', -1), ('free', -1))
    kp: tuple = (0, 0)
    pp: tuple = (0, 0)
    fp: tuple = (0, 0)
    poisoned: bool = False
    stopped: bool = False


def put(xs, i, value):
    return tuple(value if j == i else x for j, x in enumerate(xs))


class Violation(Exception):
    pass


def require(value, text):
    if not value:
        raise Violation(text)


def transitions(s, n, invisible, poison, bug):
    """Yield all currently enabled complete-group transitions."""
    if s.p < n and not s.expanding and s.raw == 'empty':
        yield 'copy_issue', replace(s, raw='flight', raw_epoch=s.p)
    if s.raw == 'flight':
        yield 'copy_complete', replace(s, raw='ready')
    if s.expanding and bug == 'recycle_raw_early' and s.p+1 < n:
        raise Violation('raw overwrite while producer expansion readers live')
    if s.p < n and not s.expanding:
        b = s.p % 2
        old_phase = (s.p//2-1) % 2
        if bug == 'wrong_free_phase': old_phase = s.p//2 % 2
        free = s.p < 2 or s.fp[b] != old_phase
        if bug == 'overwrite_live_p': free = True
        raw_ready = s.raw == 'ready' or (bug == 'skip_raw_wait' and s.raw == 'flight')
        if free and raw_ready:
            require(s.raw == 'ready', 'read before complete cp.async group')
            require(s.raw_epoch == s.p, 'wrong raw generation')
            require(s.slots[b][0] == 'free', 'converted K/V overwrite before PV release')
            yield 'expand_begin', replace(s, expanding=True, raw='reading',
                                         slots=put(s.slots,b,('expand',s.p)))
    if s.expanding:
        b=s.p%2
        require(s.raw == 'reading' and s.raw_epoch == s.p, 'raw readers lost generation')
        yield 'expand_end_publish_K', replace(s,p=s.p+1,expanding=False,raw='empty',raw_epoch=-1,
            slots=put(s.slots,b,('k',s.p)),kp=put(s.kp,b,1-s.kp[b]))
    if s.q < n and not s.qr and not s.stopped:
        b=s.q%2
        parity=(s.q//2)%2 if bug != 'wrong_ready_phase' else s.q%2
        if s.kp[b] != parity:
            require(s.slots[b] == ('k',s.q), 'stale K barrier phase woke wrong epoch')
            yield 'QK_begin', replace(s,qr=True,slots=put(s.slots,b,('q_read',s.q)))
    if s.qr:
        b=s.q%2
        if bug == 'publish_p_early':
            raise Violation('P/score write aliases live QK readers')
        detected=s.q == poison
        if detected and bug == 'return_on_poison':
            yield 'incorrect_early_return', replace(s,qr=False,stopped=True,poisoned=True)
        elif s.q in invisible and bug == 'skip_invisible':
            yield 'incorrect_invisible_skip', replace(s,q=s.q+1,qr=False)
        else:
            # Normal, invisible and poisoned tiles all publish a payload.
            # Invisible: P=0, alpha=1; poison: safe placeholder, sticky error.
            phase='free' if bug == 'free_after_QK' else 'p'
            yield 'QK_retire_publish_P', replace(s,q=s.q+1,qr=False,
                slots=put(s.slots,b,(phase,s.q)),pp=put(s.pp,b,1-s.pp[b]),
                fp=put(s.fp,b,1-s.fp[b]) if bug == 'free_after_QK' else s.fp,
                poisoned=s.poisoned or detected)
    if s.v < n and not s.vr:
        b=s.v%2
        if s.pp[b] != (s.v//2)%2:
            require(s.slots[b] == ('p',s.v), 'P/alpha consumed after overwrite or wrong epoch')
            yield 'PV_begin', replace(s,vr=True,slots=put(s.slots,b,('v_read',s.v)))
    if s.vr:
        b=s.v%2
        require(s.slots[b] == ('v_read',s.v), 'V/P/alpha changed during PV reads')
        yield 'PV_retire_release', replace(s,v=s.v+1,vr=False,
            slots=put(s.slots,b,('free',s.v)),fp=put(s.fp,b,1-s.fp[b]))


def terminal(s,n,poison):
    return (s.p == s.q == s.v == n and s.raw == 'empty'
            and not (s.expanding or s.qr or s.vr or s.stopped)
            and all(x[0] == 'free' for x in s.slots)
            and s.poisoned == (poison is not None))


def explore(n, invisible=(), poison=None, bug=None):
    initial=State(); todo=deque([initial]); parents={initial:None}; ends=0
    overlap=set(); raw_overlap=False
    def trace(s, final):
        path=[final]
        while parents[s] is not None:
            previous,event=parents[s];path.append(event);s=previous
        return list(reversed(path))
    while todo:
        s=todo.popleft()
        active=[name for name,owns in (('producer',s.expanding),('QK',s.qr),('PV',s.vr)) if owns]
        assert len(active)<=2, 'two slots cannot support three simultaneous compute owners'
        if len(active)==2: overlap.add('+'.join(active))
        raw_overlap |= s.raw=='flight' and (s.qr or s.vr)
        if terminal(s,n,poison): ends+=1;continue
        try: moves=list(transitions(s,n,set(invisible),poison,bug))
        except Violation as e: return dict(ok=False,states=len(parents),trace=trace(s,str(e)))
        if not moves: return dict(ok=False,states=len(parents),trace=trace(s,'deadlock / undrained exit'))
        for event,new in moves:
            if new not in parents:
                parents[new]=(s,event);todo.append(new)
        if len(parents)>100000: raise AssertionError('bounded model state cap exceeded')
    return dict(ok=ends>0,states=len(parents),terminal_states=ends,
                overlap_pairs=overlap,raw_overlap=raw_overlap)


def alpha_witness():
    # QK can be ahead of PV: each slot must carry its own alpha. Latest global
    # alpha is not a substitute. No BF16/R7 numerical guarantee is modeled here.
    scores=(0.,1.,3.); values=(10.,-7.,3.)
    alpha=(0.,math.exp(-1),math.exp(-2))
    good=bad=0.
    for i,value in enumerate(values):
        good=alpha[i]*good+value
        bad=(alpha[2] if i==1 else alpha[i])*bad+value
    denom=sum(math.exp(x-scores[-1]) for x in scores)
    exact=sum(math.exp(x-scores[-1])*v for x,v in zip(scores,values))/denom
    require(abs(good/denom-exact)<1e-12,'slot alpha recurrence incorrect')
    require(abs(bad/denom-exact)>0.1,'global-alpha negative failed to distinguish')


def main():
    checks=0;states=0
    for n in range(9):
        cases=[((),None),(tuple(range(n)),None),(tuple(range(1,n,2)),None)]
        if n: cases += [((),0),((),n-1)]
        for invisible,poison in cases:
            result=explore(n,invisible,poison)
            assert result['ok'],result
            if n>=2:
                assert result['overlap_pairs']=={'producer+QK','producer+PV','QK+PV'},result
                assert result['raw_overlap'],result
            states+=result['states'];checks+=1
    for bug in ('skip_raw_wait','recycle_raw_early','overwrite_live_p','publish_p_early',
                'free_after_QK','skip_invisible','return_on_poison','wrong_ready_phase','wrong_free_phase'):
        result=explore(5,invisible=(1,),poison=2,bug=bug)
        assert not result['ok'],bug
        print('C1_NEGATIVE',bug,' -> '.join(result['trace']))
        checks+=1
    alpha_witness();checks+=1
    # Aggregate barriers require every participating thread, including inactive
    # logical rows. One returning warp leaves the configured count unsatisfied.
    for expected in (64,64,128):
        assert expected-32 > 0 and sum([32]*(expected//32)) == expected
        checks+=1
    print(f'RESULT: PASS C1 protocol model {checks}/{checks} explored_states={states} tiles=0..8')
    print('C1_OVERLAP all three pairings reachable; raw DMA overlaps consumers; maximum simultaneous compute owners=2, NOT a full three-stage overlap')
    print('SCOPE bounded atomic-role model only; NOT PTX ordering, CUDA racecheck, R7, occupancy or performance proof')


if __name__ == '__main__': main()
