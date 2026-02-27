if [[ ! -d "./Result" ]]; then
 echo "Creating the directory Result"
 mkdir Result
fi

for i in {1..9}
do 
echo "====== centrality $i ======"
root -b -q Finish_v1_tof_eff.C\(${i}\)
done

# You can comment the hadd after first use
rm -rf cen13.v2_pion.root
hadd cen13.v2_pion.root cen1.v2_pion.root cen2.v2_pion.root cen3.v2_pion.root
rm -rf cen14.v2_pion.root
hadd cen14.v2_pion.root cen1.v2_pion.root cen2.v2_pion.root cen3.v2_pion.root cen4.v2_pion.root
rm -rf cen47.v2_pion.root
hadd cen47.v2_pion.root cen4.v2_pion.root cen5.v2_pion.root cen6.v2_pion.root cen7.v2_pion.root
rm -rf cen57.v2_pion.root
hadd cen57.v2_pion.root cen5.v2_pion.root cen6.v2_pion.root cen7.v2_pion.root
rm -rf cen89.v2_pion.root
hadd cen89.v2_pion.root cen8.v2_pion.root cen9.v2_pion.root

root -b -q 'Finish_v1_tof_eff.C(13)'
root -b -q 'Finish_v1_tof_eff.C(14)'
root -b -q 'Finish_v1_tof_eff.C(47)'
root -b -q 'Finish_v1_tof_eff.C(57)'
root -b -q 'Finish_v1_tof_eff.C(89)'


