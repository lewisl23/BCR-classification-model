
################################################################################
#                                                                              #
#     igBLAST analysis to analyse BCR sequences and output as AIRR format      #
#                                                                              #
################################################################################

# Get list of samples in target directory
samples=$(awk '{print $1}' sample_info.txt)

export IGDATA=/opt/ncbi-igblast-1.21.0/

for sample_id in ${samples};
do
  echo "Processing sample: ${sample_id}"

  # Run igBLAST for each sample  
  $IGDATA/bin/igblastn -query ./presto_output/${sample_id}_collapse-unique.fasta  \
  -germline_db_V $IGDATA/OGRDB/Homo_sapiens_IGH_V -germline_db_D $IGDATA/OGRDB/Homo_sapiens_IGH_D \
  -germline_db_J $IGDATA/OGRDB/Homo_sapiens_IGH_J -organism human -ig_seqtype Ig \
  -auxiliary_data $IGDATA/OGRDB/Homo_sapiens_IGH.aux -out ../OGRDB_reads/${sample_id}.airr.tsv -outfmt 19 -num_threads 50

done
