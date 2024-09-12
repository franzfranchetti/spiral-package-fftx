sig := [1..4]::Reversed([1..4]);
n := Length(sig);

#============================================================
# correctness of method

# with complex FFT
scat := Scat(fTensor(fId(n), fBase(2,0)));
fft := RC(DFT(n, -1));
diag := RC(Diag(List([0..n/2-1], i->E(2*n)^i)::List([n/2..n-1], i->-E(2*n)^i)));
ifft := 1/n * RC(DFT(n, 1));
gath := Gath(fTensor(fId(n), fBase(2,0)));
shift := gath * ifft * diag * fft * scat;
upsamplec := L(2*n, n) * VStack(I(n), shift);
upcm := MatSPL(upsamplec);

# with real FFT
Import(realdft);
rfft := PRDFT(n, -1);
twid := List([0..n/2-1], i->E(2*n)^i)::List([n/2..n-1], i->-E(2*n)^i);
twidr := Flat(List(twid{[1..n/2+1]}, e->[Re(e), Im(e)]));
twidd := RCDiag(FList(TReal, twidr));
irfft := IPRDFT(n, 1);
shiftr := 1/n * irfft * twidd * rfft;
InfinityNormMat(MatSPL(shiftr) - MatSPL(shift));
upsampler := L(2*n, n) * VStack(I(n), shiftr);
uprm := MatSPL(upsampler);
InfinityNormMat(MatSPL(upsampler) - MatSPL(upsample));

# with complex FFT 2x size
scat := Scat(fTensor(fId(n), fBase(2,0)));
fft := RC(DFT(n, -1));
zeropad := RC(DirectSum(Scat(fTensor(fBase(2,0), fId(n/2))), Scat(fTensor(fBase(2,1), fId(n/2)))));
ifft := RC(DFT(2*n, 1));
gath := Gath(fTensor(fId(2*n), fBase(2,0)));
ups2 := 1/8 * gath * ifft * zeropad * fft * scat;
ups2m := MatSPL(last);
InfinityNormMat(ups2m - uprm);

# # rotate by 1
# sigf := MatSPL(DFT(n)) * sig;
# twid := List([0..n-1], i->E(n)^i);
# sigft := List(Zip2(sigf, twid), Product);
# sigt := 1/n * MatSPL(DFT(n, -1)) * sigft;
# 
# # rotate by 2
# sigf := MatSPL(DFT(n)) * sig;
# twid := List([0..n-1], i->E(n)^(2*i));
# sigft := List(Zip2(sigf, twid), Product);
# sigt := 1/n * MatSPL(DFT(n, -1)) * sigft;
# # rotate by 3
# sigf := MatSPL(DFT(n)) * sig;
# twid := List([0..n-1], i->E(n)^(3*i));
# sigft := List(Zip2(sigf, twid), Product);
# sigt := 1/n * MatSPL(DFT(n, -1)) * sigft;

# rotate by 1/2
sigf := MatSPL(DFT(n, -1)) * sig;
twid := List([0..n/2-1], i->E(2*n)^i)::List([n/2..n-1], i->-E(2*n)^i);
sigft := List(Zip2(sigf, twid), Product);
sigt := 1/n * MatSPL(DFT(n, 1)) * sigft;
sigtc := List(sigt, ComplexAny);
sigtr := List(sigt, i->_unwrap(re(i)));

# interleave
sig2a := Flat(Zip2(sig, sigt));
sig2ac := List(sig2a, ComplexAny);
sig2ar := List(sig2a, i->_unwrap(re(i)));

#-----------------------------------
# upsample with complex FFT 2n
sig := [1..4]::Reversed([1..4]);
n := Length(sig);
sigf := MatSPL(DFT(n, -1)) * sig;
sigf2 := sigf{[1..n/2]} :: Replicate(n, 0):: sigf{[n/2+1..n]};
sig2 := 1/(n) * MatSPL(DFT(2*n, 1)) * sigf2;
sig2c := List(sig2, ComplexAny);
sig2r := List(sig2, i->_unwrap(re(i)));

#siga := List(2*[1..n], i->sig2[i]);
#sigaf := MatSPL(DFT(n, -1)) * siga;
#ediv := e -> When(e[1] = 0 and e[2] = 0, 0, e[1] / e[2]);
#tw := List(Zip2(sigaf, sigf), ediv);
#twid := List([0..n/2-1], i->E(2*n)^i)::List([n/2..n-1], i->-E(2*n)^i);

#  check correctness
sig2a = sig2;

#------------------------------------
# with RDFT
# rotate by 1/2
Import(realdft);
sigrf := MatSPL(PRDFT(n, -1)) * sig;
twid := List([0..n/2-1], i->E(2*n)^i)::List([n/2..n-1], i->-E(2*n)^i);
twidr := Flat(List(twid{[1..n/2+1]}, e->[Re(e), Im(e)]));
twidd := RCDiag(FList(TReal, twidr));
sigrft := MatSPL(twidd) * sigrf;
sigrt := 1/n * MatSPL(IPRDFT(n, 1)) * sigrft;

sigtc := List(sigrt, ComplexAny);
sigtr := List(sigrt, i->_unwrap(re(i)));

# interleave
sigar := Flat(Zip2(sig, sigtr));

# check
sigar - sig2ar;
InfinityNormMat([sigar - sig2ar]);

#############################################################
# Spiral CodeGen Script

Import(realdft);

sig := [1..4]::Reversed([1..4]);
n := Length(sig);

_noT := r -> CopyFields(r, rec(forTransposition := false));
opts := SpiralDefaults;
opts.breakdownRules.PRDFT := List([ PRDFT1_Base2, PRDFT1_CT, PRDFT_PD], _noT);
opts.breakdownRules.IPRDFT := List([ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD], _noT);
opts.breakdownRules.IPRDFT2 := List([ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT], _noT);
opts.breakdownRules.PRDFT3 := List([ PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1], _noT);


sigrf := MatSPL(PRDFT(n, -1)) * sig;
twid := List([0..n/2-1], i->E(2*n)^i)::List([n/2..n-1], i->-E(2*n)^i);
twidr := Flat(List(twid{[1..n/2+1]}, e->[Re(e), Im(e)]));

t := L(2*n, n) * VStack(I(n), 1/n * IPRDFT(n, 1) * RCDiag(FList(TReal, twidr)) * PRDFT(n, -1));

rt := RandomRuleTree(t, opts);
c := CodeRuleTree(rt, opts);
PrintCode("upsample", c, opts);

# correctness
mt := MatSPL(t);
sig2 := mt * sig;
mc := CMatrix(c, opts);
InfinityNormMat(mc - mt);



