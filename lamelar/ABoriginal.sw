# The Stillinger-Weber parameters given in the literature are pair
# specific. While most of the parameters are indeed pairwise parameters
# according to their definition, the parameters epsilon and lambda
# should be viewed as three-body dependent. Here we assume that the
# the three-body epsilon and lambda is a geometric mean of the pairwise
# epsilon and lambda.

# In lammps, the parameters for the ij pair are entered in
# the ijj three-body line. There is no unique way to convert pair
# parameters to three body parameters so the example here represents
# only one way. The three-body parameters epsilon_ijk can be calculated
# from the literature pair parameters using epsilon_ijk =
# sqrt(lambda_ij*epsilon_ij*lambda_ik*epsilon_ik)/lambda_ik, and the
# results are directly entered in this table. Obviously, this
# conversion does not change the two-body parameters epsilon_ijj. 

# The twobody ik pair parameters are entered on the i*k lines, where *
# can be any species. This is consistent with the LAMMPS requirement
# that twobody ik parameters be defined on the ikk line. Entries on all
# the other i*k lines are ignored by LAMMPS

# These entries are in LAMMPS "metal" units: epsilon = eV;
# sigma = Angstroms; other quantities are unitless

#     epsilon sigma a lambda gamma   cos(theta)     A      B     p   q  tol
A A A 1.0 1.0 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
A A B 1.0 1.0 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
A B A 1.0 1.0 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
A B B 1.4 1.15 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
B B B 1.0 1.0 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
B B A 1.0 1.0 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
B A B 1.0 1.0 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
B A A 1.4 1.15 1.80 0 0  -0.3333333333   7.049556277 0.6022245584  4.0  0.0  0.0
