
################################################################################
#                                                                              #
#     igBLAST analysis to analyse BCR sequences adn out oput as AIRR format    #
#                                                                              #
################################################################################

# Get list of samples in target directory
samples=$(awk '{print $1}' sample_info.txt)

export IGDATA=/home/s2106664/dissertation/ncbi-igblast-1.22.0

for sample_id in ${samples}; do
  echo "Processing sample: ${sample_id}"

  # Run igBLAST for each sample  
  $IGDATA/bin/igblastn -query ./presto_output/${sample_id}_collapse-unique.fasta  \
  -germline_db_V $IGDATA/igblastdb/airr_c_human_ig.V -germline_db_D $IGDATA/igblastdb/airr_c_human_igh.D \
  -germline_db_J $IGDATA/igblastdb/airr_c_human_ig.J -organism human -ig_seqtype Ig \
  -auxiliary_data $IGDATA/optional_file/human_gl.aux -out ${sample_id}.airr.tsv -outfmt 19 -num_threads 50

done