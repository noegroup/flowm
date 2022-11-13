#!/usr/bin/bash

mkdir -p downloaded
cd downloaded
# downloading the ala2 (capped alanine) data set from Noe group FTP
wget -c https://ftp.mi.fu-berlin.de/pub/cmb-data/ala2_cg_2fs_Hmass_2_HBonds.npz -O ala2_cg_data.npz
wget -c https://ftp.mi.fu-berlin.de/pub/cmb-data/ala2_cg.pdb -O ala2_cg.pdb

# downloading the CLN (capped chignolin) data set from Noe group FTP
wget -c https://ftp.mi.fu-berlin.de/pub/cmb-data/cln_charmm22star_basic_force_aggr_CA_MSM_weight.npz -O cln_ca_data.npz
wget -c https://ftp.mi.fu-berlin.de/pub/cmb-data/cln_ca.pdb -O cln_ca.pdb

cd ..
