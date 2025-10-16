_conjEvenIPRDFT := function(n, k)
    local ii, _rcdf1, _rcdf2, r12, spl;

    ii := Ind((n-2)/2);

    _rcdf1 := RCData(diagAdd(diagMul(fConst(TComplex, n/2-1, Cplx(0, 1)), fCompose(dOmega(n, k), fAdd(n/2, n/2-1, 1))), fConst(TReal, n/2-1, 1)));
    _rcdf2 := diagMul(RCData(diagAdd(fCompose(dOmega(n, k), fAdd(n/2, n/2-1, 1), J(n/2-1)), fConst(TComplex, n/2-1, E(4)))), 
        diagTensor(fConst(TReal, n/2-1, 1), FList(TReal, [1, -1])));

    r12 := F(2) * Gath(fStack(fBase(n+2,0), fBase(n+2, n)));
    spl := 
        VStack(r12, 
            ISum(ii, (n-2)/2, 
                Scat(fTensor(fBase(ii), fId(2))) * _HStack(I(2), I(2)) * VStack( 
                    RCDiag(fCompose(fPrecompute(_rcdf1), fTensor(fBase(ii), fId(2)))) * Gath(fCompose(fAdd(n+2, n-2, 2), fTensor(fBase(ii), fId(2)))), 
                    RCDiag(fCompose(fPrecompute(_rcdf2), fTensor(fBase(ii), fId(2)))) * J(2) * Gath(fCompose(fAdd(n+2, n-2, 2), fTensor(fCompose(J(n/2-1), fBase(ii)), fId(2)))) 
            ))
        );
    return spl;
end;

_conjEvenPRDFT := function(N, rot)
    local ii, d2af, d2bf, conjd, spl;

    ii := Ind((N-2)/2);
    
    d2af := diagAdd(diagMul(fConst(TComplex, N/2-1, 1/2 * Cplx(0, -1)), fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1))), fConst(TReal, N/2-1, 1/2));
    d2bf := diagAdd(diagMul(fConst(TComplex, N/2-1, -1/2 * Cplx(0, -1)), fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1))), fConst(TReal, N/2-1, 1/2));
    conjd := diagTensor(fConst(TReal, (N-2)/2, 1), FList(TReal, [1, -1]));
    spl := VStack(
        DirectSum(
            Blk([[1,1],[0,0]]),
            ISum(ii, (N-2)/2, 
                Scat(fTensor(fBase(ii), fId(2))) * _HStack(I(2), I(2)) * VStack(
                    RCDiag(RCData(fCompose(fPrecompute(d2af), fBase(ii)))) * Gath(fTensor(fBase(ii), fId(2))),
                    RCDiag(RCData(fCompose(fPrecompute(d2bf), fBase(ii)))) * Diag(FList(TReal, [1, -1])) * Gath(fTensor(fCompose(fTensor(J((N-2)/2)), fBase(ii)), fId(2))) 
                )
            )
        ),
        Blk([[1,-1], [0,0]]) * Gath(fTensor(fBase(N/2,0), fId(2)))
    );
    return spl;
end;


NewRulesFor(PRDFT1, rec(
    PRDFT1_NR := rec(
        applicable := nt -> IsEvenInt(nt.params[1]) and not nt.hasTags(),
        children  := nt -> [[ DFT(nt.params[1]/2, nt.params[2]) ]],
        apply := (nt, C, cnt) -> _conjEvenPRDFT(nt.params[1], nt.params[2]) * RC(C[1])
    )
));

NewRulesFor(IPRDFT1, rec(
    IPRDFT1_NR := rec(
        applicable := nt -> IsEvenInt(nt.params[1]) and not nt.hasTags(),
        children  := nt -> [[ DFT(nt.params[1]/2, nt.params[2]) ]],
        apply := (nt, C, cnt) -> RC(C[1]) * _conjEvenIPRDFT(nt.params[1], nt.params[2])
    )
));
