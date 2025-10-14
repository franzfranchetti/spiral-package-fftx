
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
