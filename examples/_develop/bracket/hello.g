ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

n := 4;
x1 := [1..4];
x2 := [5..8];

t1 := MatSPL(DFT(n, -1));

y1 := t1*x1;
y2 := t1*x2;

z := List(Zip2(y1, y2), Product);
t2 := 1/n * MatSPL(DFT(n, 1));

y := t2 * z;

#==================================================================

x1fs := x1::Replicate(n, 0);
xf2s := x2::Replicate(n, 0);

t1a := MatSPL(DFT(2*n, -1)); 
y1fs := t1a*x1fs;
y2fs := t1a*x2fs;

zfs := List(Zip2(y1fs, y2fs), Product);
t2a := 1/n * MatSPL(DFT(2*n, -1));

y2 := t2a * zfs;

gath := Gath(fTensor(fBase(2,1), fId(n)));
gath2 := HStack(O(n,n), I(n));
InfinityNormMat(MatSPL(gath) - MatSPL(gath2));

y2fs := MatSPL(gath) * y2;
