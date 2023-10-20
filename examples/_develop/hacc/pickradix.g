knownFactors := [];


factorInto := function(N, stages)
    local stageval, fct, n, mapping, factors, m, buckets, j, sad, mn, idx, bestFct;
    stageval := exp(log(N)/stages).v;
    
    fct := Factors(N);
    n := Length(fct);
    mapping := ApplyFunc(Cartesian, Replicate(n, [1..stages]));
    
    factors := [];
    for m in mapping do
        buckets := Replicate(stages, 1);
        for j in [1..n] do
            buckets[m[j]] := buckets[m[j]] * fct[j];
           Add(factors, buckets);
        od;
    od;
    
    sad := List(factors, m -> Sum(List(m, i -> AbsFloat(i - stageval))));
    mn := Minimum(sad);
    idx := Position(sad, mn);
    bestFct := factors[idx];

    return bestFct;
end;

bestFactors := function(N, max_factor)
    local factors, i, f, bestf;
    
    if IsBound(knownFactors[N]) then return knownFactors[N]; fi;
    
    factors := List([2..4], i -> factorInto(N, i));
    
    bestf := Filtered(factors, f -> ForAll(f, i -> i < 26))[1];
    knownFactors[N] := bestf;
    return bestf;
end;




N := 30000;

factors := bestFactors(N, 16);

