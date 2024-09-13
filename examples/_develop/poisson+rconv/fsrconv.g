#=============================================================================
# Real Free Space Convolution

ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));

# construct the Toeplitz matrix and the filter taps
n := 8;
range := [1..2*n];
filt := List(range, i->Random([1..100]));
symbf := 1/(2*n) * MatSPL(DFT(2*n, -1)) * filt;
tfilt := filt{[n+2..2*n]}::filt{[1]}::filt{[2..n]};
cfilt := Reversed(tfilt{[1..n]});

conv := Gath(fTensor(fBase(2,0), fId(n))) * DFT(2*n, 1) * Diag(symbf) * DFT(2*n, -1) * Scat(fTensor(fBase(2,0), fId(n)));
convm := MatSPL(conv);
#convt := Toeplitz(Reversed(convm[1])::Drop(TransposedMat(convm)[1],1));
convt := Toeplitz(tfilt);
InfinityNormMat(convm - MatSPL(convt));

# break free space convolutrion into sum of 2 convolutions of size n
diag1 := Diag(symbf{[1..n] * 2 - 1});
diag2 := Diag(symbf{[1..n] * 2});

dft1 := DFT(n, -1);
dft2 := DFT(n, -1) * Diag(List([0..n-1], i -> (-E(2*n))^((n-1)*i)));
idft1 := DFT(n, 1);
idft2 := Diag(List([0..n-1], i -> E(2*n)^i)) * DFT(n, 1);

step1 := idft1 * diag1 * dft1;
step2 := idft2 * diag2 * dft2;
convd := SUM(step1, step2);
InfinityNormMat(MatSPL(convd) - convm);

# Toeplitz = Circulant + Skew-Circulant
filt1 := MatSPL(step1)[1];
filt1a := MatSPL(DirectSum(I(1), J(n-1)))*MatSPL(idft1)*MatSPL(RowVec(symbf{[1..n] * 2 - 1}))[1];
filt1b:= List([1]::Reversed([2..n]), i->filt[i]+filt[i+n])/2;
InfinityNormMat([filt1 - filt1a]);
InfinityNormMat([filt1 - filt1b]);
step1c := Circulant(filt1);
InfinityNormMat(MatSPL(step1) - MatSPL(step1c));

filt2 := MatSPL(step2)[1];
filt2a := cfilt - filt1a;
InfinityNormMat([filt2 - filt2a]);
step2c := SkewCirculant(filt2);
InfinityNormMat(MatSPL(step2) - MatSPL(step2c));

# fast algorithm for step1
drf := Flat(List(symbf{[1..n/2+1] * 2 - 1}, a->[Re(a), Im(a)]));
rcdiag1 := RCDiag(FList(TReal, drf));
rdft1 := PRDFT(n, -1);
irdft1 := IPRDFT(n, 1);
rstep1 := irdft1 * rcdiag1 * rdft1;
InfinityNormMat(MatSPL(step1) - MatSPL(rstep1));

# fast algorithm for step2
m := n/2;
q := 1/(Sqrt(2))*VStack(
    HStack(Mat([[Sqrt(2)]]), O(1, n-1)),
    HStack(O(n/2-1,1), I(n/2-1), O(n/2-1,1), J(n/2-1)),
    HStack(O(1, n/2), Mat([[Sqrt(2)]]), O(1, n/2-1)),
    HStack(O(n/2-1,1), -J(n/2-1), O(n/2-1,1), I(n/2-1))
);

scale := Diag([1/Sqrt(2)]::Replicate(m-1, 1));
qtut := DirectSum(scale * DCT2(m), scale * J(m)*DST2(m)*J(m));
sigmaq := DirectSum(I(m), J(m)) * L(n,2);
sq := sigmaq.transpose() * qtut.transpose() * q.transpose();
sigmat2 := 2/m * Sqrt(2) * Tensor(I(m), Diag(FList(TInt, [1,-1]))) * sq;

ut2 := MatSPL(sigmat2) * filt2;
sigmad2 := RCDiag(FList(TReal, ut2));

rstep2 := sq.transpose() * sigmad2 * sq;
InfinityNormMat(MatSPL(step2) - MatSPL(rstep2));

# full algorithm = real step1 + real step2
rconv := SUM(rstep1, rstep2);
InfinityNormMat(MatSPL(rconv) - convm);
