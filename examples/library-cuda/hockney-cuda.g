
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Load(fftx);
ImportAll(fftx);
Debug(true);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

nn := List([4..8], i->2^i);

i := 3;
n := nn[i];
d := 3;
n := Replicate(d, n);

ns := List(n, i->i/2);
nd := ns;

name := "rfsconv3d";
symvar := var("symbl", TPtr(TReal));

t := TFCall(
        Compose([
            ExtractBox(n, [[0..nd[1]-1],[0..nd[2]-1],[0..nd[3]-1]]),
            IMDPRDFT(n, 1),
            RCDiag(FDataOfs(symvar, 2*n[1]*n[2]*(n[3]/2+1), 0)),
            MDPRDFT(n, -1), 
            ZeroEmbedBox(n, [[0..ns[1]-1],[0..ns[2]-1],[0..ns[3]-1]])]),
        rec(fname := name, params := [symvar])
    );

opts := conf.getOpts(t);
tt := opts.tagIt(t);

c := opts.fftxGen(tt);
opts.prettyPrint(c);

PrintTo(name::".cu", opts.prettyPrint(c));
