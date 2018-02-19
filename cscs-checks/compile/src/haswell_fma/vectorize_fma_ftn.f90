double precision a, b, c, d, x, y, z
DIMENSION A(1000000), B(1000000), C(1000000), D(1000000)
READ*, X, Y, Z
A = LOG(X); B = LOG(Y); D = LOG(Z); C = A * B + D
PRINT*, C(500000)
END
