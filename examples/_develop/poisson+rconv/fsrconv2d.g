#=============================================================================
# Real 2D Free Space Convolution without Domain Doubling

ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));
RotDiag := lst -> RCDiag(FList(TReal, List(Zip2(lst, Flat(Replicate(Length(lst)/2, [1, -1]))), Product)));

#==============================================================================
# problem setup
n := 4;
range := [1..n^2];
filt := List(range, i->Random([1..100]));

#==============================================================================
# Fast Real 1D Free Space Convolution without Domain Doubling

# transform for step1
rdft1 := PRDFT(n, 1);
irdft1 := IPRDFT(n, -1);

# transform for step2
m := n/2;
dct := DCT2(m);
dst := DST2(m);

scale := Diag([1/Sqrt(2)]::Replicate(m-1, 1));
q := 1/Sqrt(2) * VStack(
    HStack(Mat([[Sqrt(2)]]), O(1, n-1)),
    HStack(O(n/2-1,1), I(n/2-1), O(n/2-1,1), J(n/2-1)),
    HStack(O(1, n/2), Mat([[Sqrt(2)]]), O(1, n/2-1)),
    HStack(O(n/2-1,1), -J(n/2-1), O(n/2-1,1), I(n/2-1))
);
qtut := DirectSum(scale * dct, scale * J(m) * dst * J(m));
sigmaq := DirectSum(I(m), J(m)) * L(n,2);
dtt := Compose(q, qtut, sigmaq).transpose();
idtt := 2/m * Sqrt(2) * dtt.transpose();

#------------------------------------------------------------------------------
# 2D operators

dtts := [rdft1, dtt];
idtts := [irdft1, idtt];

dtt2d := List(Cartesian(dtts, dtts), i->ApplyFunc(Tensor, i));
idtt2d := List(Cartesian(idtts, idtts), i->ApplyFunc(Tensor, i));

# filter taps -- that's probably not right
taps := List(dtt2d, i->MatSPL(i)*filt);
rcds := List(taps, RotDiag);

# 2d rconv
convs := List([1..4], i-> idtt2d[i] * rcds[i] * dtt2d[i]);
conv2d := ApplyFunc(SUM, convs);
