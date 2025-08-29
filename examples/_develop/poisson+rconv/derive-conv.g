ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));
RotDiag := lst -> RCDiag(FList(TReal, List(Zip2(lst, Flat(Replicate(Length(lst)/2, [1, -1]))), Product)));

# set up the problem
n := 4;
filt := [-n..n-1];
filt := List(filt, i->Random([1..100]));

# baseline
symbf := 1/(2*n) * MatSPL(DFT(2*n, -1)) * filt;
conv := Gath(fTensor(fBase(2,0), fId(n))) * DFT(2*n, 1) * Diag(symbf) * DFT(2*n, -1) * Scat(fTensor(fBase(2,0), fId(n)));
convm := MatSPL(conv);

# the two components
dft1 := DFT(n, -1);
dft2 := DFT(n, -1) * Diag(List([0..n-1], i -> (-E(2*n))^((n-1)*i)));
idft1 := DFT(n, 1);
idft2 := Diag(List([0..n-1], i -> E(2*n)^i)) * DFT(n, 1);

diagc := Diag(symbf)^L(2*n, n);
pconv2 := Tensor(RowVec(1,1), I(n)) * DirectSum(idft1, idft2) * diagc * VStack(dft1, dft2);
pconv2m := MatSPL(pconv2);
pconv2m = pconvm;

# break free space convolutrion into sum of 2 convolutions of size n
diag1 := Diag(symbf{[1..n] * 2 - 1});
diag2 := Diag(symbf{[1..n] * 2});
diagcc := DirectSum(diag1, diag2);
MatSPL(diagcc) = MatSPL(diagc);

step1 := idft1 * diag1 * dft1;
step2 := idft2 * diag2 * dft2;
conv3 := SUM(step1, step2);
conv3m := MatSPL(conv3);
conv3m = convm;

pm(step1);
pm(step2);
#===========================
step2m := MatSPL(step2);
scstep2 := SkewCirculant(MatSPL(step2)[1]);
InfinityNormMat(MatSPL(scstep2) - step2m);

rcstep2 := RC(idft2 * diag2 * dft2);
scat := Scat(fTensor(fId(n), fBase(2,0)));
gath := Gath(fTensor(fId(n), fBase(2,0)));
rstep2 := gath*rcstep2*scat;
InfinityNormMat(MatSPL(rstep2)-step2m);

rstep2a := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2 * diag2 * dft2) * Scat(fTensor(fId(n), fBase(2,0)));
InfinityNormMat(MatSPL(rstep2a)-step2m);

rstep2b := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2) * RC(diag2) * RC(dft2) * Scat(fTensor(fId(n), fBase(2,0)));
InfinityNormMat(MatSPL(rstep2b)-step2m);
#--------------------

rdft2 := RC(dft2)* Scat(fTensor(fId(n), fBase(2,0)));
ridft2 := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2);
rdiag2 := RC(diag2);

rstep2 := ridft2 * rdiag2 * rdft2;
InfinityNormMat(MatSPL(rstep2) - step2m);
#--------------------

symb2a := Flat(List(symbf{[1..n] * 2}, i-> [Re(i), Im(i)]));
rcdiag2a := RCDiag(FList(TReal, rsymb2a));
InfinityNormMat(MatSPL(rcdiag2a) - MatSPL(RC(diag2)));

rstep2a := ridft2 * rcdiag2a * rdft2;
InfinityNormMat(MatSPL(rstep2a) - step2m);

#--------------------
dft2b := DFT(n, 1) * Diag(List([0..n-1], i -> (-E(2*n))^((n-1)*i)));
rdft2b := RC(dft2b)* Scat(fTensor(fId(n), fBase(2,0)));
idft2b := Diag(List([0..n-1], i -> E(2*n)^i)) * DFT(n, -1);
ridft2b := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2b);

symb2b := Flat(List(symbf{[1..n] * 2}, i-> [Re(i), Im(i)]));
rcdiag2b := RotDiag(symb2b);
InfinityNormMat(MatSPL(rcdiag2b) - MatSPL(RC(diag2)));

rstep2a := ridft2b * rcdiag2b * rdft2b;
InfinityNormMat(MatSPL(rstep2b) - step2m);


















#-------------
#dft2a := DFT(n, -1) * Diag(List([0..n-1], i -> (-E(2*n))^((n-1)*i)));
#idft2a := Diag(List([0..n-1], i -> E(2*n)^i)) * DFT(n, 1);
rsymb2a := Flat(List(symbf{[1..n] * 2}, i-> [Re(i), Im(i)]));
rcrdiag2a := RCDiag(FList(TReal, rsymb2a));
InfinityNormMat(MatSPL(rcrdiag2a) - MatSPL(rdiag2));


