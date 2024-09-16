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
rstep2 := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2 * diag2 * dft2)* Scat(fTensor(fId(n), fBase(2,0)));

rdft2 := RC(dft2)* Scat(fTensor(fId(n), fBase(2,0)));
ridft2 := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2);
rdiag2 := RC(diag2);
rstep2m := MatSPL(ridft2 * rdiag2 * rdft2);
rstep2m = step2m;

#----
dft2a := DFT(n, 1) * Diag(List([0..n-1], i -> (-E(2*n))^((n-1)*i)));
idft2a := Diag(List([0..n-1], i -> E(2*n)^i)) * DFT(n, -1);
rsymb2 := Flat(List(symbf{[1..n] * 2}, i-> [Re(i), Im(i)]));
rcdiag2a := RotDiag(rsymb2);
rstep2 := Gath(fTensor(fId(n), fBase(2,0))) * RC(idft2a) * rcdiag2a * RC(dft2a)* Scat(fTensor(fId(n), fBase(2,0)));
