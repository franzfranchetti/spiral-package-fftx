
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Import(fftx.platforms.cuda);
Import(fftx.codegen);
Import(simt);

Class(BatchFFTCUDADeviceOpts, FFTXCUDADeviceOpts, rec(
#    tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTBlockDimX, ASIMTLoopDim() ],
#    tags := [ASIMTGridDimX, ASIMTBlockDimX, ASIMTKernelFlag(ASIMTLoopDim()) ],
#    tags := [ASIMTBlockDimZ, ASIMTKernelFlag(ASIMTBlockDimX), ASIMTKernelFlag(ASIMTBlockDimY) ],
    tags := [ASIMTKernelFlag(ASIMTGridDimY), ASIMTGridDimX, ASIMTBlockDimZ],

    operations := rec(Print := s -> Print("<FFTX Batch FFT CUDA Device options record>"))    
));

batchFftCUDADeviceOpts := function(arg) # specific to FFT size 100...
    local opts;
    
    opts := Copy(BatchFFTCUDADeviceOpts);
    opts.breakdownRules.Circulant := [Circulant_PRDFT_FDataNT];
    opts.breakdownRules.PRDFT := List([PRDFT1_Base1, PRDFT1_Base2, CopyFields(PRDFT1_CT, 
            rec(allChildren := P ->Filtered(PRDFT1_CT.allChildren(P), i->When(P[1] = 100, Cols(i[1]) = 4, true)))), 
        PRDFT_PD], _noT);
    opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD ], _noT);
    opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT ], _noT);
    opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT ], _noT);
    opts.breakdownRules.DFT := [ DFT_Base, 
        CopyFields(DFT_CT, rec(children := nt ->Filtered(DFT_CT.children(nt), i->When(nt.params[1] = 100, Cols(i[1]) = 4, true)))), 
        DFT_PD ];

    opts.breakdownRules.TTensorInd := [TTensorInd_SIMT];    
    opts.breakdownRules.TL := [L_SIMT];
    opts.breakdownRules.TIterHStack := [TIterHStack_SIMT];
    opts.breakdownRules.TIterVStack := [TIterVStack_SIMT];
    opts.breakdownRules.MDPRDFT := [MDPRDFT_3D_SIMT];
    opts.breakdownRules.IMDPRDFT := [IMDPRDFT_3D_SIMT];
    opts.breakdownRules.TRC := [TRC_SIMT];
    opts.breakdownRules.MDDFT := [ MDDFT_Base, CopyFields(MDDFT_tSPL_RowCol, rec(switch := true)), MDDFT_RowCol, MDDFT_RowCol_3D_SIMT];
    opts.breakdownRules.TTensor := [ AxI_IxB, IxB_AxI ];
    opts.breakdownRules.TTensorI := [ IxA_SIMT,  AxI_SIMT, trivialLoop ];
    opts.breakdownRules.TSparseMat := [CopyFields(TSparseMat_base, rec(max_rows := 1)), TSparseMat_VStack];
    
    opts.globalUnrolling := 23;
    opts.codegen.GathPtr := fftx.codegen.MultiPtrCodegenMixin.GathPtr;
    opts.codegen.ScatPtr := fftx.codegen.MultiPtrCodegenMixin.ScatPtr;
    opts.codegen.OO := (self, o, y, x, opts) >> skip();
    
    opts.sumsgen.IterHStack := MultiPtrSumsgenMixin.IterHStack;
    opts.preProcess := t -> ApplyStrategy(t, 
                    [ MergedRuleSet(RulesFuncSimp, RulesSums), 
                      RulesFFTXPromoteNT, 
                      MergedRuleSet(RulesFuncSimp, RulesSums) ],
        BUA, opts);
        
    opts.postProcessSums := (s, opts) -> let(s1 := ApplyStrategy(s, [ MergedRuleSet(RulesFuncSimp, RulesSums, RulesSIMTFission) ], BUA, opts),
        FixUpCUDASigmaSPL(s1, opts)); 

#    opts.useDeref := false;
    # for debugging...
#    opts.generateInitFunc := false;
#    opts.codegen := DefaultCodegen;
#    opts.arrayBufModifier := "";
#    opts.arrayDataModifier := "static const";
    opts.max_threads := 1024;
        
    return opts;
end;


confBatchFFTCUDADevice := rec(
    defaultName := "confBatchFFTCUDADevice",
    defaultOpts := (arg) >> rec(useBatchFFTCUDADevice := true),
    confHandler := batchFftCUDADeviceOpts 
);

fftx.FFTXGlobals.registerConf(confBatchFFTCUDADevice);



