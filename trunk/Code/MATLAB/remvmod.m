P = primes(50);
n = 1e8;
disp('mod')
tic
for p = P
    disp(p)
    for i=1:n
        mod(i,p);
    end
end
modtime = toc;
disp('rem')
tic
for p = P
    disp(p)
    for i=1:n
        rem(i,p);
    end
end
remtime = toc;