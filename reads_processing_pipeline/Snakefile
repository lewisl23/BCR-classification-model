
################################################################################
#                                                                              #
#     preprocessing pipeline to process raw reads and output as AIRR format    #
#                                                                              #
################################################################################

rule assemble:
    input:
        read_1 = "raw_reads/{sample}_1.fastq"
        read_2 = "raw_reads/{sample}_2.fastq"
    output:
        pass_file = "presto/presto_output/{sample}_assemble-pass.fastq"
    container:
        "docker://ghcr.io/lewisl23/presto-0.7.4:latest"
    shell: 
        """

        AssemblePairs.py align -1 {input.read_1} -2 {input.read_2} \
        --coord illumina --rc tail --outname {wildcards.sample} \
        --outdir presto/presto_output --log presto/presto_output/{wildcards.sample}_AP.log

        # Print FAIL message if reads does not pass assembly
        if [ ! -f {output.pass_file}]; then
            echo "Assembly failed for sample {wildcards.sample}" >&2
            exit 1
        fi

        """

rule filter:
    input:
        "presto/presto_output/{sample}_assemble-pass.fastq"
    output:
        pass_file = "presto/presto_output/{sample}_quality-pass.fastq"
    container:
        "docker://ghcr.io/lewisl23/presto-0.7.4:latest"
    shell:
        """

        FilterSeq.py quality -s {input} -q 20 --outname {wildcards.sample} \
        --outdir presto/presto_output --log presto/presto_output/{wildcards.sample}_FS.log

        # Print FAIL message if reads does not pass filtering
        if [ ! -f {output.pass_file}]; then
            echo "Filtering failed for sample {wildcards.sample}" >&2
            exit 1
        fi

        """

rule mask_primer:
    input:
        reads = "presto/presto_output/{sample}_quality-pass.fastq"
        primer = "presto/Cowan_CPrimers.fasta"
    output:
        pass_file = "presto/presto_output/{sample}_primers-pass.fastq"
    container:
        "docker://ghcr.io/lewisl23/presto-0.7.4:latest"
    shell:
        """

        MaskPrimers.py score -s {input.reads} -p {input.primer} \
        --start 0 --maxerror 0 --mode tag --revpr --pf CPRIMER \
        --outname {wildcards.sample} --outdir presto/presto_output \
        --log presto/presto_output/{wildcards.sample}_MP.log 
        
        # Print FAIL message if reads does not pass filtering
        if [ ! -f {output.pass_file}]; then
            echo "Primer masking failed for sample {wildcards.sample}" >&2
            exit 1
        fi

        """

rule collapse_unique:
    input:
        "presto/presto_output/{sample}_primers-pass.fastq"
    output:
        "presto/presto_output/{sample}_collapse-unique.fasta"
    container:
        "docker://ghcr.io/lewisl23/presto-0.7.4:latest"
    shell:
        """

        CollapseSeq.py -s {input} -n 20 --inner --uf CPRIMER --fasta \
        --outname {wildcards.sample} --outdir presto/presto_output \
        --log presto/presto_output/{wildcards.sample}_CS.log

        """

rule igblast:
    input:
        "presto/presto_output/{sample}_collapse-unique.fasta"
    output:
        "blast_results/{sample}.airr.tsv"
    container:
        "docker://ghcr.io/lewisl23/igblast-1.21.0:latest"
    shell:
        """

        echo "Processing sample: {wildcards.sample}"

        # Run igBLAST for each sample  
        $IGDATA/bin/igblastn -query presto/presto_output/{wildcards.sample}_collapse-unique.fasta  \
        -germline_db_V $IGDATA/database/Homo_sapiens_IGH_V \
        -germline_db_D $IGDATA/database/Homo_sapiens_IGH_D \
        -germline_db_J $IGDATA/database/Homo_sapiens_IGH_J \
        -organism human -ig_seqtype Ig \
        -auxiliary_data $IGDATA/database/Homo_sapiens_IGH.aux \
        -out blast_results/{wildcards.sample}.airr.tsv -outfmt 19

        """

rule all:
    input:
        expand("blast_results/{sample}.airr.tsv", sample=config["samples"])