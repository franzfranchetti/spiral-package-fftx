
# SPIRAL script generated by MdrconvSolver
# Thu Jun 02 13:07:10 2022

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.confGPU();

t := let(symvar := var("sym", TPtr(TReal)),
    TFCall(
        Compose([
            ExtractBox([162,162,162], [[81..161],[81..161],[81..161]]),
            IMDPRDFT([162,162,162], 1),
            RCDiag(FDataOfs(symvar, 4304016, 0)),
            MDPRDFT([162,162,162], -1),
            ZeroEmbedBox([162,162,162], [[0..80],[0..80],[0..80]])
        ]),
        rec(fname := "Mdrconv_81_cu", params := [symvar])
    )
);

opts := conf.getOpts(t);
opts.wrapCFuncs := true;
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
PrintTo("Mdrconv_81_cu.icode", c);
PrintTo("Mdrconv_81_cu.cu", opts.prettyPrint(c));
