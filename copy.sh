#!bin/sh

files="Fukushima"
part1="abnormal"
part2="unobvious"

outfiles=data
outfile="${files}_${part1}_${part2}.txt"
outfile1="${files}_${part1}_${part2}1.txt"
if [ -f $outfile ]
then
rm $outfile
fi
if [ -f $outfile1 ]
then
rm $outfile1
fi
echo "datanum" "eventnum" "eventname" "mag" "fc" "fvar" "label1" "label2" "spefile" "stffile" > ${outfile} 
echo "eventnum" "eventname" "mag" "fc" "fvar" "label1" "label2" > ${outfile1} 
infile=${files}/${part1}_${part2}/fval_ave.txt

enum=`wc $infile | awk '{print $1}'`
i=1
datanum=0
#enum=1
while [ $i -le $enum ]
do
ename=`awk -v s=$i '{if(NR == s)print $1}' $infile`
if [ -d data/$ename ]
then
rm -rf data/$ename
fi
mkdir data/$ename
mag=`awk -v s=$i '{if(NR == s)print $2}' $infile`
fc=`awk -v s=$i '{if(NR == s)print $3}' $infile`
fvar=`awk -v s=$i '{if(NR == s)print $4}' $infile`
label1=${part1}
label2=${part2}
echo $i ${ename} ${mag} ${fc} ${fvar} ${label1} ${label2} >> ${outfile1}

ls ${files}/${part1}_${part2}/spe/${ename}/*.txt | awk -F '/' '{print $5}' > spe_list.txt
ls ${files}/${part1}_${part2}/stf/${ename}/*.txt | awk -F '/' '{print $5}' > stf_list.txt
dnum=`wc stf_list.txt | awk '{print $1}'`
j=1
while [ $j -le $dnum ]
do
spefile=`awk -v s=$j '{if(NR == s)print $1}' spe_list.txt`
stffile=`awk -v s=$j '{if(NR == s)print $1}' stf_list.txt`
cp ${files}/${part1}_${part2}/spe/${ename}/${spefile} data/${ename}/
cp ${files}/${part1}_${part2}/stf/${ename}/${stffile} data/${ename}/
datanum=`expr $datanum + 1`
echo $datanum $i ${ename} ${mag} ${fc} ${fvar} ${label1} ${label2} ${spefile} ${stffile} >> ${outfile}
j=`expr $j + 1`
done

i=`expr $i + 1`
done
rm stf_list.txt spe_list.txt
