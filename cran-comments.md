## Resubmission

Version 0.2.1 addresses the comments by Gregor Seyer. Thank you for taking the time to review this submission.

Changes to the code:

* T and F replaced by TRUE and FALSE throughout.

* no messages to console by default; they can be displayed by directly setting `verbose=TRUE`.

Changes to the docs:

* removed `rm(list=ls())` in the example.

* removed `\dontrun{}` from the examples. Currently the toy example for `spamtree` seems to work in under 5s on the test environments listed below (no notes about that).


## Test environments
* local: Ubuntu 18.04.5 LTS, R 4.1.0
* win-builder (devel and release)
* r-hub (Ubuntu Linux 20.04.1 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran; Debian Linux, R-devel, GCC ASAN/UBSAN)

## R CMD check results
There were no ERRORs or WARNINGs. 

There was 1 NOTE:

* checking installed package size ... NOTE
    installed size is 11.1Mb
    sub-directories of 1Mb or more:
      libs  10.9Mb

There was and additional 1 NOTE on Fedora/Ubuntu on R-hub. 

(Words are correctly spelled and the link is valid on my local computer.)

* Possibly mis-spelled words in DESCRIPTION:
  Dunson (8:188)
  Peruzzi (8:176)
  SpamTrees (8:120)

* Found the following (possibly) invalid URLs:
  URL: https://doi.org/10.1093/biomet/asp078
    From: man/CrossCovarianceAG10.Rd
    Status: 403
    Message: Forbidden
    
## Downstream dependencies
There are no downstream dependencies
