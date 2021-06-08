\name{CrossCovarianceAG10}
\alias{CrossCovarianceAG10}
\title{
Multivariate non-separable cross-covariance function on latent domain of variables.
}
\description{
This function implements the cross-covariance function used in Peruzzi and Dunson (2021), which is derived from eq. 7 in Apanasovich and Genton (2010).
}
\usage{
CrossCovarianceAG10(coords1, mv1, coords2, mv2, 
                    ai1, ai2, phi_i, thetamv, Dmat)
}
\arguments{
  \item{coords1}{
matrix with spatial coordinates
}
  \item{mv1}{
integer vector with variable IDs. The length must match the number of rows in \code{coords1}
}
  \item{coords2}{
matrix with spatial coordinates
}
  \item{mv2}{
integer vector with variable IDs. The length must match the number of rows in \code{coords2}
}
  \item{ai1}{
\eqn{q}-dimensional vector
}
  \item{ai2}{
\eqn{q}-dimensional vector
}
  \item{phi_i}{
\eqn{q}-dimensional vector
}
  \item{thetamv}{
for bivariate data (\eqn{q=2}), this is a scalar. For \eqn{q>2}, this is a vector with elements \eqn{\alpha, \beta, \phi}.
}
  \item{Dmat}{
symmetric matrix of dimension \eqn{(q, q)} with zeroes on the diagonal and whose \eqn{(i,j)} element is \eqn{\delta_{i,j}}.
}

}
\details{
Suppose we have \eqn{q} variables. For \eqn{h>0} and \eqn{\Delta > 0} define:
\deqn{C(h, \Delta) = \frac{ \exp \{ -\phi \|h \| / \exp \{\beta \log(1+\alpha \Delta)/2 \} \} }{ \exp \{ \beta \log(1 + \alpha \Delta) \}  } }
and for \eqn{j=1, \dots, q}, define \eqn{C_j(h) = \exp \{ -\phi_j \|h \| \}}. 

Then the cross-covariance between the \eqn{i}th margin of a \eqn{q}-variate process \eqn{w(\cdot)} at spatial location \eqn{s} and the \eqn{j}th margin at location \eqn{s'} is built as follows. For \eqn{i = j} as
\deqn{Cov(w(s, \xi_i), w(s', \xi_j)) = \sigma_{i1}^2 C(h, 0) + \sigma_{i2}^2 C_i(h) ,}
whereas if \eqn{i \neq j} it is defined as
\deqn{Cov(w(s, \xi_i), w(s', \xi_j)) = \sigma_{i1} \sigma_{i2} C(h, \delta_{ij}),}
where \eqn{\xi_i} and \eqn{\xi_j} are the latent locations of margin \eqn{i} and \eqn{j} in the domain of variables and \eqn{\delta_{ij} = \| \xi_i - \xi_j \|} is their distance in such domain.
}
\value{
The cross-covariance matrix for all pairwise locations.
}
\references{
Apanasovich, T. V. and Genton, M. G. (2010) Cross-covariance functions for multivariate random fields based on latent dimensions. \emph{Biometrika}, 97:15-30. \doi{10.1093/biomet/asp078}

Peruzzi, M. and Dunson, D. B. (2021) Spatial Multivariate Trees for Big Data Bayesian Regression. \url{https://arxiv.org/abs/2012.00943}
}
\author{
Michele Peruzzi \email{michele.peruzzi@duke.edu}
}

\examples{

library(magrittr)
library(dplyr)
library(spamtree)

SS <- 10
xlocs <- seq(0.0, 1, length.out=SS)
coords <- expand.grid(xlocs, xlocs)
c1 <- coords \%>\% mutate(mv_id=1)
c2 <- coords \%>\% mutate(mv_id=2)

coords <- bind_rows(c1, c2)
coords_q <- coords \%>\% dplyr::select(-mv_id)
cx <- coords_q \%>\% as.matrix()
mv_id <- coords$mv_id

ai1 <- c(1, 1.5)
ai2 <- c(.1, .51)
phi_i <- c(1, 2)
thetamv <- 5

q <- 2
Dmat <- matrix(0, q, q)
Dmat[2,1] <- 1
Dmat[upper.tri(Dmat)] <- Dmat[lower.tri(Dmat)]

CC <- CrossCovarianceAG10(cx, mv_id, cx, mv_id, ai1, ai2, phi_i, thetamv, Dmat)


}
