_useOmega := false;

_conjEvenIPRDFT := function(n, k)
    local ii, _rcdf1, _rcdf2, r12, spl, j;

    ii := Ind((n-2)/2);
    j := Ind(n-2);

    if _useOmega then
        _rcdf1 := RCData(diagAdd(diagMul(fConst(TComplex, n/2-1, V(Cplx(0, 1))), fCompose(dOmega(n, k), fAdd(n/2, n/2-1, 1))), fConst(TReal, n/2-1, V(1))));
        _rcdf2 := diagMul(RCData(diagAdd(fCompose(dOmega(n, k), fAdd(n/2, n/2-1, 1), J(n/2-1)), fConst(TComplex, n/2-1, V(E(4))))), 
            diagTensor(fConst(TReal, n/2-1, 1), FList(TReal, [1, -1])));
    else            
        _rcdf1 := Lambda(j, cond(eq(0, imod(j, 2)),
            V(1)-sinpi(k*fdiv(tcast(TDouble, idiv(j,V(2))+V(1)),tcast(TDouble, V(n/2)))),
             cospi(k*fdiv(tcast(TDouble, idiv(j-V(1), V(2))+V(1)),tcast(TDouble, V(n/2))))
            ));
        _rcdf2 := Lambda(j, cond(eq(0, imod(j, 2)), 
            V(-1)*cospi(k*fdiv(tcast(TDouble, idiv(j,V(2))+V(1)),tcast(TDouble, V(n/2)))),
            V(-1)-sinpi(k*fdiv(tcast(TDouble, idiv(j-V(1), V(2))+V(1)),tcast(TDouble, V(n/2))))
            ));
    fi;    

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
    local ii, d2af, d2bf, spl, j;

    ii := Ind((N-2)/2);
    j := Ind(N-2);
#Error();

    if _useOmega then
        d2af := RCData(diagAdd(diagMul(fConst(TComplex, N/2-1, V(1/2 * Cplx(0, -1))), fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1))), fConst(TReal, N/2-1, 1/2)));
        d2bf := RCData(diagAdd(diagMul(fConst(TComplex, N/2-1, V(-1/2 * Cplx(0, -1))), fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1))), fConst(TReal, N/2-1, 1/2)));
    else
        d2af := Lambda(j, cond(eq(0, imod(j, 2)), 
            V(1/2)*(V(1)+sinpi(rot*fdiv(tcast(TDouble, idiv(j,V(2))+V(1)), tcast(TDouble, V(N/2))))),
            V(-1/2)*(cospi(rot*fdiv(tcast(TDouble, j+V(1)), tcast(TDouble, V(N)))))
        ));
        d2bf := Lambda(j, cond(eq(0, imod(j, 2)), 
            V(1/2)*(V(1)-sinpi(rot*fdiv(tcast(TDouble, idiv(j,V(2))+V(1)), tcast(TDouble, V(N/2))))),
            V(1/2)*(cospi(rot*fdiv(tcast(TDouble, j+V(1)), tcast(TDouble, V(N)))))
        ));
        
    fi;
    spl := VStack(
        DirectSum(
            Blk([[1,1],[0,0]]),
            ISum(ii, (N-2)/2, 
                Scat(fTensor(fBase(ii), fId(2))) * _HStack(I(2), I(2)) * VStack(
                    RCDiag(fCompose(fPrecompute(d2af), fTensor(fBase(ii), fId(2)))) * Gath(fTensor(fBase(ii), fId(2))),
                    RCDiag(fCompose(fPrecompute(d2bf), fTensor(fBase(ii), fId(2)))) * Diag(FList(TReal, [1, -1])) * Gath(fTensor(fCompose(fTensor(J((N-2)/2)), fBase(ii)), fId(2))) 
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
