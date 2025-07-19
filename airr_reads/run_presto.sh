
################################################################################
#                                                                              #
#  presto workflow used to assemble, filter (quality control, and mask primer) #
#                                                                              #
################################################################################

# Get list of samples in target directory
samples=$(awk '{print $1}' sample_info.txt)

source activate presto

for sample_id in ${samples}; do
    echo "Processing sample: ${sample_id}"
    
    # Assemble paired-end reads
    AssemblePairs.py align -1 ${sample_id}_1.fastq -2 ${sample_id}_2.fastq --coord illumina --rc tail --nproc 50 --outname ${sample_id} --outdir ./presto_output --log ${sample_id}_AP.log
    
    # Filter sequences to remove low quality reads (<20)
    FilterSeq.py quality -s ./presto_output/${sample_id}_assemble-pass.fastq -q 20 --outname ${sample_id} --outdir ./presto_output --log ${sample_id}_FS.log --nproc 50
    
    # Mask primers using a primer file
    MaskPrimers.py score -s ./presto_output/${sample_id}_quality-pass.fastq -p Cowan_CPrimers.fasta --start 0 --maxerror 0 --mode tag --revpr --pf CPRIMER --nproc 50 --log ${sample_id}_MP.log --outname ${sample_id} --outdir ./presto_output
    
    # Collapse sequences to remove duplicates
    CollapseSeq.py -s ./presto_output/${sample_id}_primers-pass.fastq -n 20 --inner --uf CPRIMER --fasta --outname ${sample_id} --outdir ./presto_output
    
    # ParseHeaders table
    #ParseHeaders.py table -s ./presto_output/${sample_id}_collapse-unique.fastq -f ID DUPCOUNT CPRIMER VPRIMER --outdir ./presto_output
done

conda deactivate
