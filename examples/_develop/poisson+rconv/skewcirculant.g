ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));

m := 16;
n := 2*m;

filt := List([1..n], i-> Random([1..100]));
#filt := [1..n];
sf := SkewCirculant(filt);

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

sigma2 := (2/m)^2 * qtut.transpose()* qt * sf * q * qtut;
InfinityNormMat(MatSPL(sigma2) - MatSPL(sigma));

s :=  q * qtut * sigma * qtut.transpose() * qt;
sc := SkewCirculant(MatSPL(s)[1]);

InfinityNormMat(MatSPL(sc) - MatSPL(s));
InfinityNormMat(MatSPL(sc) - MatSPL(sf));

#=======================================
# now deriving sigma

# alpha := [1..m];
# beta := [m+1..2*m];
# sigma := VStack(
#     HStack(Diag(alpha), J(Length(beta))*Diag(beta)),
#     HStack(-J(Length(beta))*Diag(Reversed(beta)), Diag(Reversed(alpha))));

sigmam := MatSPL(sigma);
alpha := List([1..m], i-> sigmam[i][i]);
beta := List([1..m], i-> sigmam[i][2*m-i+1]);
sigmax := VStack(
    HStack(Diag(alpha), J(Length(beta))*Diag(Reversed(beta))),
    HStack(-J(Length(beta))*Diag(beta), Diag(Reversed(alpha))));

InfinityNormMat(MatSPL(sigma) - MatSPL(sigmax));

s1 := MatSPL(s)[1];
sigmat := Sqrt(2) * qtut.transpose()* qt;
ut := MatSPL(sigmat) * s1;

alpha2 := 2/m*ut{[1..m]};
beta2 := 2/m*Reversed(ut{[m+1..2*m]});

sigma2 := VStack(
    HStack(Diag(alpha2), J(Length(beta2))*Diag(Reversed(beta2))),
    HStack(-J(Length(beta2))*Diag(beta2), Diag(Reversed(alpha2))));

InfinityNormMat(MatSPL(sigma2) - MatSPL(sigma));

