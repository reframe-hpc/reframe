# Getting Started

ReFrame is split into two parts. The frontend and the backend. The frontend is used to execute the regression test and the backand is used to ...

## TL;DR &ndash; Running the latest release on Piz Daint

### [v2.1] Run as Jenkins (user `jenscscs`)

```bash
su - jenscscs
module load PyRegression/2.1
reframe --reservation=maintenance -t maintenance -r | tee $APPS/UES/jenkins/regression/maintenance/reports/$(date +%FT%T).txt
```
