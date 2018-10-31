#include<stdio.h>
#include<math.h>

void func1(double *a, double *b, double *c, double *d){
    int i;
    for (i=0; i<1000000; i++){
        c[i]=a[i]*b[i]+d[i];
    }
}

int main(){
    double a[1000000], b[1000000], c[1000000], d[1000000], x, y, z;
    int i;
    if (scanf("%le %le %le", &x, &y, &z) < 3) {
        fprintf(stderr, "too few arguments\n");
        return 1;

    }
    for (i=0; i<1000000; i++){
        a[i]=log(x);
        b[i]=log(y);
        d[i]=log(z);
    }
    void (*func1_ptr)(double *a, double *b, double *c, double *d) = func1;
    func1_ptr(a, b, c, d);
    printf("%e\n", b[50000]);
    return 0;
}
