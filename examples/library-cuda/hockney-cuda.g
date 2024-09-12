
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

#n := 64;
#n := 128;
#n := 256;
n := 3*64;

ns := n/2;
nd := n/2;

PrintLine("hockney-mlc-cuda: n = ", n, " nd = ", nd, " ns = ", ns, ";\t\t##PICKME##");

t := let(name := "hockney"::StringInt(n)::"_"::StringInt(nd)::"_"::StringInt(ns), 
        symvar := var("symbl", TPtr(TReal)),
    TFCall(
        Compose([
#            ExtractBox([n,n,n], [[n-nd..n-1],[n-nd..n-1],[n-nd..n-1]]),
            ExtractBox([n,n,n], [[0..nd-1],[0..nd-1],[0..nd-1]]),
            IMDPRDFT([n,n,n], 1),
            RCDiag(FDataOfs(symvar, 2*n*n*(n/2+1), 0)),
            MDPRDFT([n,n,n], -1), 
            ZeroEmbedBox([n,n,n], [[0..ns-1],[0..ns-1],[0..ns-1]])]),
        rec(fname := name, params := [symvar])
    )
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintLine("hockney-cuda: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");
