To use in a Jupyter notebook, first run
`from data_conversion import conversion`

Then sample usage:

    geno = conversion.get_geno_data()
    geno = conversion.drop_single_value_cols(geno)
    geno_binary = conversion.convert_geno_to_binary(geno)
    geno_ternary = conversion.convert_geno_to_ternary(geno)

for genotype data, and:

    pheno = conversion.get_pheno_data()
for phenotype data.

