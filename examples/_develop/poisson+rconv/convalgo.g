n:=16;
vec := [1..n];
vec := List(vec, i->Random([1..100]));
vec2 := vec::Replicate(n,0);

# forward FFT
vecf := MatSPL(DFT(n, -1)) * vec;
vec2f := MatSPL(DFT(2*n, -1)) * vec2;
vec2f{[1..n]*2-1} = vecf;
vecff := vec2f{[1..n]*2};

vecc := 1/n * MatSPL(DFT(n, 1)) * vecff;
scale := List([1..n], i->vecc[i]/vec[i]);
scale2 := List([0..n-1], i -> (-E(2*n))^((n-1)*i));
scale = scale2;

#------------------
v1 := MatSPL(DFT(n, -1)) * vec;
v2 := MatSPL(DFT(n, -1)) * List(Zip2(scale, vec), Product);
vv :=  Flat(Zip2(v1, v2));
vv = vec2f;

#==================
# inverse FFT
ivecf := [1..2*n];
ivecf := List(ivecf, i->Random([1..100]));

ivec := MatSPL(DFT(2*n, 1)) * ivecf;
ivecl := ivec{[1..n]};

ivec1 := MatSPL(DFT(n, 1)) * (ivecf{[1..n]*2-1});
ivec2 := MatSPL(DFT(n, 1)) * (ivecf{[1..n]*2});

#------------------
iscale := List([1..n], i->(ivecl[i] - ivec1[i])/ivec2[i]);
iscale2 := List([0..n-1], i -> E(2*n)^i);
iscale = iscale2;

ivecl2 := ivec1 + List(Zip2(iscale, ivec2), Product);
ivecl = ivecl2;

#=============================================================================
# convolution

# zeropadding
n := 16;
filt := [-n..n-1];
filt := List(filt, i->Random([1..100]));

symbf := 1/(2*n) * MatSPL(DFT(2*n, -1)) * filt;
conv := Gath(fTensor(fBase(2,0), fId(n))) * DFT(2*n, 1) * Diag(symbf) * DFT(2*n, -1) * Scat(fTensor(fBase(2,0), fId(n)));
convm := MatSPL(conv);

# non-zero padding DFT/iDFT
dft1 := DFT(n, -1);
dft2 := DFT(n, -1) * Diag(List([0..n-1], i -> (-E(2*n))^((n-1)*i)));
pdft := L(2*n, n) * VStack(dft1, dft2);
pdft1 := DFT(2*n, -1) * Scat(fTensor(fBase(2,0), fId(n)));
MatSPL(pdft) = MatSPL(pdft1);

idft1 := DFT(n, 1);
idft2 := Diag(List([0..n-1], i -> E(2*n)^i)) * DFT(n, 1);
ipdft := Tensor(RowVec(1,1), I(n)) * DirectSum(idft1, idft2) * L(2*n, 2);
ipdft1 := Gath(fTensor(fBase(2,0), fId(n))) * DFT(2*n, 1);
MatSPL(ipdft) = MatSPL(ipdft1);

# non-zero padding convolution
pconv := ipdft * Diag(symbf) * pdft;
pconvm := MatSPL(pconv);
pconvm = convm;

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
pconv3 := SUM(step1, step2);
pconv3m := MatSPL(pconv3);
pconv3m = pconvm;

pm(step1);
pm(step2);

# real DFT
ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

# step 1
drf := Flat(List(symbf{[1..n/2+1] * 2 - 1}, a->[Re(a), Im(a)]));
rcdiag1 := RCDiag(FList(TReal, drf));
rdft1 := PRDFT(n, -1);
irdft1 := IPRDFT(n, 1);
rstep1 := irdft1 * rcdiag1 * rdft1;
InfinityNormMat(MatSPL(step1) - MatSPL(rstep1));

step1c := Circulant(MatSPL(step1)[1]);
InfinityNormMat(MatSPL(step1) - MatSPL(step1c));

# step2
SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));

filt := MatSPL(step2)[1];
step2c := SkewCirculant(filt);
InfinityNormMat(MatSPL(step2) - MatSPL(step2c));
InfinityNormMat((MatSPL(step1c + step2c) - pconvm));

n := Length(filt);
m := n/2;

q := 1/(Sqrt(2))*VStack(
    HStack(Mat([[Sqrt(2)]]), O(1, n-1)),
    HStack(O(n/2-1,1), I(n/2-1), O(n/2-1,1), J(n/2-1)),
    HStack(O(1, n/2), Mat([[Sqrt(2)]]), O(1, n/2-1)),
    HStack(O(n/2-1,1), -J(n/2-1), O(n/2-1,1), I(n/2-1))
);

qt := q.transpose();
scale := Diag([1/Sqrt(2)]::Replicate(m-1, 1));
qtut := DirectSum(scale * DCT2(m), scale * J(m)*DST2(m)*J(m));

s1 := filt;
sigmat := Sqrt(2) * qtut.transpose()* qt;
ut := MatSPL(sigmat) * s1;

alpha := 2/m*ut{[1..m]};
beta := 2/m*Reversed(ut{[m+1..2*m]});

sigma := VStack(
    HStack(Diag(alpha), J(Length(beta))*Diag(Reversed(beta))),
    HStack(-J(Length(beta))*Diag(beta), Diag(Reversed(alpha))));

sigma2 := (2/m)^2 * qtut.transpose()* qt * step2c * q * qtut;
InfinityNormMat(MatSPL(sigma2) - MatSPL(sigma));

rstep2 :=  q * qtut * sigma * qtut.transpose() * qt;
InfinityNormMat(MatSPL(step2) - MatSPL(rstep2));

# full algorithm = real step1 + real step2
prconv := SUM(rstep1, rstep2);
InfinityNormMat(MatSPL(prconv) - pconvm);




