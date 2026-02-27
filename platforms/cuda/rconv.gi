NewRulesFor(PrunedMDPRDFT, rec(
    PrunedMDPRDFT_tSPL_Base := rec(
        applicable     := nt -> Length(nt.params[1]) = 1,

        children       := nt -> [[ PrunedPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        apply          := (nt, C, Nonterms) -> C[1]
    ),

    PrunedMDPRDFT_tSPL_RowCol1 := rec(
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)),
                              PrunedPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])) ]],

        apply := (nt, C, cnt) ->  RC(Tensor(C[1], I(C[2].dims()[1]/2))) * Tensor(I(Product(List(cnt[1].params[4], i->Length(i)))), C[2])
    )
));


NewRulesFor(PrunedIMDPRDFT, rec(
    PrunedIMDPRDFT_tSPL_Base := rec(
        applicable     := nt -> Length(nt.params[1]) = 1,

        children       := nt -> [[ PrunedIPRDFT(nt.params[1][1], nt.params[2]).withTags(nt.getTags()) ]],

        apply          := (nt, C, Nonterms) -> C[1]
    ),

    PrunedIMDPRDFT_tSPL_RowCol1 := rec(
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children  := nt -> [[ PrunedIPRDFT(Last(nt.params[1]), nt.params[3], 1, Last(nt.params[2])),
                              PrunedIMDDFT(DropLast(nt.params[1], 1), nt.params[3], 1, DropLast(nt.params[2], 1)) ]],

        apply := (nt, C, cnt) -> Tensor(I(Product(List(cnt[2].params[4], i->Length(i)))), C[1]) * RC(Tensor(C[2], I(C[1].dims()[2]/2)))
    )
));


NewRulesFor(PrunedMDDFT, rec(
    PrunedMDDFT_tSPL_Base := rec(
        info := "PrunedMDDFT -> PrunedDFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ PrunedDFT(P[1][1], P[2], P[3], P[4][1]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    ),
    PrunedMDDFT_tSPL_RowCol := rec (
        info := "PrunedMDDFT_n -> PrunedMDDFT_n/d, PrunedMDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            pats := nt.params[4],
            List([1..len-1],
            i -> [ PrunedMDDFT(dims{[1..i]}, nt.params[2], nt.params[3], pats{[1..i]}), 
                   PrunedMDDFT(dims{[i+1..len]}, nt.params[2], nt.params[3], pats{[i+1..len]}) ])),

        apply := (nt, C, Nonterms) -> let(
            n1 := Cols(Nonterms[1]),
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), C[2])
        )
    )
));

NewRulesFor(PrunedIMDDFT, rec(
    PrunedIMDDFT_tSPL_Base := rec(
        info := "PrunedIMDDFT -> PrunedIDFT",
        applicable     := nt -> Length(nt.params[1])=1,
        children       := nt -> let(P := nt.params, tags := nt.getTags(), [[ PrunedIDFT(P[1][1], P[2], P[3], P[4][1]).withTags(tags) ]]),
        apply          := (nt, C, Nonterms) -> C[1]
    ),
    PrunedIMDDFT_tSPL_RowCol := rec (
        info := "PrunedIMDDFT_n -> PrunedIMDDFT_n/d, PrunedIMDDFT_d",
        applicable := nt -> Length(nt.params[1]) > 1 and not nt.hasTags(),

        children := nt -> let(
            dims := nt.params[1],
            len := Length(dims),
            pats := nt.params[4],
            List([1..len-1],
            i -> [ PrunedIMDDFT(dims{[1..i]}, nt.params[2], nt.params[3], pats{[1..i]}), 
                   PrunedIMDDFT(dims{[i+1..len]}, nt.params[2], nt.params[3], pats{[i+1..len]}) ])),

        apply := (nt, C, Nonterms) -> let(
            n1 := Cols(Nonterms[1]),
            n2 := Rows(Nonterms[2]),
            Tensor(C[1], I(n2)) *
            Tensor(I(n1), C[2])
        )
    )
));

NewRulesFor(IOPrunedMDRConv, rec(
    IOPrunedMDRConv_tSPL_InvDiagFwd := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            [[ TCompose([ 
                                PrunedIMDPRDFT(nt.params[1], nt.params[4], 1),
                                RCDiag(fCompose(diagMul(fConst(TReal, nt.params[2].domain(),1/Product(nt.params[1])), 
                                    nt.params[2]))),
                                PrunedMDPRDFT(nt.params[1], nt.params[6], -1)]).withTags(nt.getTags())
                            ]]),

       apply := (nt, C, cnt) -> C[1]
    ),

    IOPrunedMDRConv_tSPL_5stage := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 

       children  := nt -> let( nlist := nt.params[1],
                               diag := nt.params[2],
                               oblk := nt.params[3],
                               opats := nt.params[4],
                               iblk := nt.params[5],
                               ipats := nt.params[6],
                               nn := Product(nlist),
                               nz := nlist[1],
                               ny := nlist[2],
                               nx := nlist[3],
                               nxf := nx/2+1,
                               i := Ind(nxf),
                               j := Ind(ny),                               
                               hfunc := fCompose(diagMul(fConst(TReal, nz, 1/nn), CRData(diag)), fTensor(fId(nz), fBase(j), fBase(i))),

                               prdft1d := PrunedPRDFT(nx, -1, iblk, ipats[3]),      # stage 1: PRDFT x
                               pdft1d := PrunedDFT(ny, -1, iblk, ipats[2]),         # stage 2: DFT y
                               iopconv := IOPrunedConv(nz, hfunc, oblk, opats[1], iblk, ipats[1], true), # stage 3+4+5: complex conv in z
                               ipdft1d := PrunedIDFT(ny, 1, oblk, opats[2]),        # stage 6: iDFT in y
                               iprdft1d := PrunedIPRDFT(nx, 1, oblk, opats[3]),     # stage 7: iPRDFT in x
                               nlist := nt.params[1],
                               nxs := iblk * Length(ipats[3]),
                               nys := iblk * Length(ipats[2]),
                               nzs := iblk * Length(ipats[1]),
                               nxd := oblk * Length(opats[3]),
                               nyd := oblk * Length(opats[2]),
                               nzd := oblk * Length(opats[1]),
                               stage1 := TTensorI(prdft1d, nzs*nys, APar, APar),
                               stage2 := TRC(TTensorI(TTensorI(pdft1d, nxf, AVec, AVec), nzs, APar, APar)),
                               stage543 := TRC(TCompose([TL(nzd*ny*nxf, nzd, 1 ,1), 
                                    TTensorInd(TTensorInd(iopconv, i, APar, APar), j, APar, APar), TL(nzs*ny*nxf, nxf*ny, 1, 1)])),
                               stage6 := TRC(TTensorI(TTensorI(ipdft1d, nxf, AVec, AVec), nzd, APar, APar)),
                               stage7 := TTensorI(iprdft1d, nzd*nyd, APar, APar), 
                                    
                               conv3dr := TCompose([stage7, stage6, stage543, stage2, stage1]),
#                               Error(),
                               
                               [[ conv3dr.withTags(nt.getTags()) ]]),
#                               [ [ TCompose([ TGrp(TCompose([
#                                             TTensorI(PrunedIPRDFT(nx, 1, oblk, opats[3]), 
#                                                Product(List(DropLast(nt.params[2], 1), Length)), APar, APar),
#                                             TL(cdim * Product(List(DropLast(nlist, 1), Length)) / 2, Product(List(DropLast(nlist, 1), Length)), 1, 2), 
#                                       ])) ] ::
#                                       Reversed(List([1..Length(nlist)-1], j->let(i := nlist[j], 
#                                           DropLast(nlist, 1), TRC(TTensorI(PrunedIDFT(i, 1, oblk, opats[j]), 
#                                                cdim * Product(nlist{[j+1..Length(nt.params[2])-1]}) * Product(List(nlist{[1..j]}, Length))/(i), 
#                                           APar, AVec)))))
#                                      ::
#                                      TDiag(nt.params[2])
#                                      ::
#                                       List([1..Length(nt.params[1])-1], j->
#                                        let(i := nt.params[1][j], TRC(TTensorI(PrunedDFT(i, a_exp, 1, nt.params[2][j]), 
#                                            rcdim * Product(nt.params[1]{[j+1..Length(nt.params[2])-1]}) * Product(List(nt.params[2]{[1..j]}, Length))/(i), 
#                                            AVec, APar)))) ::
#                                               [ TGrp(TCompose([TL(rcdim * Product(List(DropLast(nt.params[2], 1), Length)) / 2, rcdim / 2, 1, 2), 
#                                                 TTensorI(PrunedPRDFT(Last(a_lengths), a_exp, 1, Last(nt.params[2])), 
#                                                    Product(List(DropLast(nt.params[2], 1), Length)), APar, APar)
#                                                 ])) ]).withTags(tags) ]] ),

        apply := (nt, C, cnt) -> C[1]
    ),
    

    # Hockney–Eastwood streamed FFT algorithm, xyz
    IOPrunedMDRConv_3D_5step := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nn := Product(nlist),
                            nz := nlist[1],
                            ny := nlist[2],
                            nx := nlist[3],
                            nxf := nx/2+1,
                            i := Ind(nxf),
                            j := Ind(ny),
                            hfunc := fCompose(diagMul(fConst(TReal, nz, 1/nn), CRData(diag)), fTensor(fId(nz), fBase(j), fBase(i))),
                            [[ PrunedPRDFT(nx, -1, iblk, ipats[3]),  # stage 1: PRDFT x
                                PrunedDFT(ny, -1, iblk, ipats[2]),    # stage 2: DFT y
                                IOPrunedConv(nz, hfunc, oblk, opats[1], iblk, ipats[1], true), # stage 3+4+5: complex conv in z
                                PrunedIDFT(ny, 1, oblk, opats[2]), # stage 6: iDFT in y
                                PrunedIPRDFT(nx, 1, oblk, opats[3]),   # stage 7: iPRDFT in x
                                InfoNt([i, j])
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    pdft1d := C[2],
                                    iopconv := C[3],
                                    ipdft1d := C[4],
                                    iprdft1d := C[5],
                                    i := cnt[6].params[1][1],
                                    j := cnt[6].params[1][2],
                                    nlist := nt.params[1],
                                    nn := Product(nlist),
                                    nz := nlist[1],
                                    ny := nlist[2],
                                    nx := nlist[3],
                                    nxf := nx/2+1,
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    nxs := iblk * Length(ipats[3]),
                                    nys := iblk * Length(ipats[2]),
                                    nzs := iblk * Length(ipats[1]),
                                    nxd := oblk * Length(opats[3]),
                                    nyd := oblk * Length(opats[2]),
                                    nzd := oblk * Length(opats[1]),
                                    stage1 := Tensor(I(nzs*nys), prdft1d),
                                    stage2 := RC(Tensor(I(nzs), pdft1d, I(nxf))),
                                    stage543 := RC(L(nzd*ny*nxf, nzd) * IDirSum(j, IDirSum(i, iopconv)) * L(nzs*ny*nxf, nxf*ny)),
                                    stage6 := RC(Tensor(I(nzd), ipdft1d, I(nxf))),
                                    stage7 := Tensor(I(nzd*nyd), iprdft1d),
                                    
                                    conv3dr := stage7 * stage6 * stage543 * stage2 * stage1,
                                    conv3dr
                            )
    ),
#==============================================================================
# FIgure out how to flip the tensor products for the CUDA/SIMT case above

    IOPrunedMDRConv_tSPL_5stage_flip := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 

       children  := nt -> let( nlist := nt.params[1],
                               diag := nt.params[2],
                               oblk := nt.params[3],
                               opats := nt.params[4],
                               iblk := nt.params[5],
                               ipats := nt.params[6],
                               nn := Product(nlist),
                               nz := nlist[1],
                               ny := nlist[2],
                               nx := nlist[3],
                               nxf := nx/2+1,
                               i := Ind(nxf),
                               j := Ind(ny),                               
                               hfunc := fCompose(diagMul(fConst(TReal, nz, 1/nn), CRData(diag)), fTensor(fId(nz), fBase(i), fBase(j))),

                               prdft1d := PrunedPRDFT(nx, -1, iblk, ipats[3]),      # stage 1: PRDFT x
                               pdft1d := PrunedDFT(ny, -1, iblk, ipats[2]),         # stage 2: DFT y
                               iopconv := IOPrunedConv(nz, hfunc, oblk, opats[1], iblk, ipats[1], true), # stage 3+4+5: complex conv in z
                               ipdft1d := PrunedIDFT(ny, 1, oblk, opats[2]),        # stage 6: iDFT in y
                               iprdft1d := PrunedIPRDFT(nx, 1, oblk, opats[3]),     # stage 7: iPRDFT in x
                               nlist := nt.params[1],
                               nxs := iblk * Length(ipats[3]),
                               nys := iblk * Length(ipats[2]),
                               nzs := iblk * Length(ipats[1]),
                               nxd := oblk * Length(opats[3]),
                               nyd := oblk * Length(opats[2]),
                               nzd := oblk * Length(opats[1]),
                               
                               stage7 := TGrp(TCompose([TTensorI(iprdft1d, nzd*nyd, APar, APar), TL(nzd*nyd*nxf, nzd*nyd, 1, 2)])),  

#                               stage6 := TGrp(TRC(TCompose([TTensorI(TTensorI(ipdft1d, nxf, APar, APar), nzd, APar, APar), TL(nzd*ny*nxf, nxf, 1, 1)]))),
#                               stage543 := TGrp(TRC(TCompose([TL(nzd*ny*nxf, nzd, 1 ,1), 
#                                    TTensorInd(TTensorInd(iopconv, i, APar, APar), j, APar, APar), TL(nzs*ny*nxf, nxf*ny, 1, 1)]))),
#                               stage2 := TGrp(TRC(TCompose([TL(nzs*ny*nxf, nzs*ny, 1, 1), TTensorI(pdft1d, nxf*nzs, APar, APar)]))),

                               stage6 := TRC(TTensorI(ipdft1d, nxf*nzd, APar, APar)),
                               stage543 := TGrp(TRC(TCompose([TL(nzd*ny*nxf, nxf*nzd, 1 ,1), 
                                    TTensorInd(TTensorInd(iopconv, j, APar, APar), i, APar, APar), TL(nzs*ny*nxf, ny, 1, 1)]))),
                               stage2 := TRC(TTensorI(pdft1d, nxf*nzs, APar, APar)),

                               stage1 := TGrp(TCompose([TL(nzs*nys*nxf, nxf, 1, 2), TTensorI(prdft1d, nzs*nys, APar, APar)])),
                                    
                               conv3dr := TCompose([stage7, stage6, stage543, stage2, stage1]),
                               
                               [[ conv3dr.withTags(nt.getTags()) ]]
        ),

        apply := (nt, C, cnt) -> C[1]
    ),

    # Hockney–Eastwood streamed FFT algorithm, xyz
    IOPrunedMDRConv_3D_5step_flip := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> nt.hasTags() and Length(nt.params[1]) = 3 and IsFunc(nt.params[7]) and nt.params[7]()
                                    and nt.params[3] = 1 and nt.params[5] = 1, 
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nn := Product(nlist),
                            nz := nlist[1],
                            ny := nlist[2],
                            nx := nlist[3],
                            nxf := nx/2+1,
                            i := Ind(nxf),
                            j := Ind(ny),
                            hfunc := fCompose(diagMul(fConst(TReal, nz, 1/nn), CRData(diag)), fTensor(fId(nz), fBase(i), fBase(j))),
                            [[ PrunedPRDFT(nx, -1, iblk, ipats[3]),  # stage 1: PRDFT x
                                PrunedDFT(ny, -1, iblk, ipats[2]),    # stage 2: DFT y
                                IOPrunedConv(nz, hfunc, oblk, opats[1], iblk, ipats[1], true), # stage 3+4+5: complex conv in z
                                PrunedIDFT(ny, 1, oblk, opats[2]), # stage 6: iDFT in y
                                PrunedIPRDFT(nx, 1, oblk, opats[3]),   # stage 7: iPRDFT in x
                                InfoNt([i, j])
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    pdft1d := C[2],
                                    iopconv := C[3],
                                    ipdft1d := C[4],
                                    iprdft1d := C[5],
                                    i := cnt[6].params[1][1],
                                    j := cnt[6].params[1][2],
                                    nlist := nt.params[1],
                                    nn := Product(nlist),
                                    nz := nlist[1],
                                    ny := nlist[2],
                                    nx := nlist[3],
                                    nxf := nx/2+1,
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    nxs := iblk * Length(ipats[3]),
                                    nys := iblk * Length(ipats[2]),
                                    nzs := iblk * Length(ipats[1]),
                                    nxd := oblk * Length(opats[3]),
                                    nyd := oblk * Length(opats[2]),
                                    nzd := oblk * Length(opats[1]),
                                    stage7 := Tensor(I(nzd*nyd), iprdft1d) * RC(L(nzd*nyd*nxf, nzd*nyd)),

#                                    stage6 := RC(Tensor(I(nxf*nzd), ipdft1d)* L(nzd*ny*nxf, nxf)),
#                                    stage543 := RC(L(nzd*ny*nxf, nzd) * IDirSum(j, IDirSum(i, iopconv)) * L(nzs*ny*nxf, nxf*ny)),
#                                    stage2 := RC(L(nzs*ny*nxf, nzs*ny) * Tensor(I(nxf*nzs), pdft1d)),

                                    stage6 := RC(Tensor(I(nxf*nzd), ipdft1d)),
                                    stage543 := RC(L(nzd*ny*nxf, nxf*nzd) * IDirSum(i, IDirSum(j, iopconv)) * L(nzs*ny*nxf, ny)),
                                    stage2 := RC(Tensor(I(nxf*nzs), pdft1d)),

                                    stage1 := RC(L(nzs*nys*nxf, nxf)) * Tensor(I(nzs*nys), prdft1d),
                                    conv3dr := stage7 * stage6 * stage543 * stage2 * stage1, 
                                    #Error(),
                                    conv3dr
                            )
    )
));
