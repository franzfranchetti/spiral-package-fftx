
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
n := [64, 64, 256];

ns := [n[1]/2, n[2]/2, n[3]/2];
nd := [n[1]/2, n[2]/2, n[3]/2];


t := let(name := "hockney", 
        symvar := var("symbl", TPtr(TReal)),
    TFCall(
        Compose([
            ExtractBox(n, [[0..nd[1]-1],[0..nd[2]-1],[0..nd[3]-1]]),
            IMDPRDFT(n, 1),
            RCDiag(FDataOfs(symvar, 2*n[1]*n[2]*(n[3]/2+1), 0)),
            MDPRDFT(n, -1), 
            ZeroEmbedBox(n, [[0..ns[1]-1],[0..ns[2]-1],[0..ns[3]-1]])]),
        rec(fname := name, params := [symvar])
    )
);

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintLine("hockney-cuda: codegen test only (no compiled test with 'symbol')\t\t##PICKME##");
