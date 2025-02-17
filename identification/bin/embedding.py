# %%
import os 
import warnings
warnings.filterwarnings('ignore')
from .HmmFea import RunHMM, ReadHmm, GenerateTrans
from .EdpFea import *
from .Hexamer import ReadLogScore
from .ProtParam import param
from .ORF import ExtractORF

''' Prepare feature calculation input '''
def pre(filename, lnc_seq, threads=1):
    lnc_seq_dict = {record.id: str(record.seq) for record in lnc_seq}
    ## translate fasta to protein and HMMER
    tmpdir = './identification/tmp/'
    output_prefix = os.path.basename(filename).split('.')[0]
    tmp_protein_file = os.path.join(tmpdir, output_prefix + "_protein.fasta")
    GenerateTrans(filename, tmp_protein_file)
    tmp_hmmer_prefix = os.path.join(tmpdir, output_prefix + "_hmmer")
    tmp_hmmer_out = os.path.join(tmpdir, output_prefix + "_hmmer.out2")
    pfam = "./identification/src/Pfam-A.hmm"
    RunHMM(tmp_protein_file, tmp_hmmer_prefix, pfam, thread=threads)

    logscore_dict = ReadLogScore('./identification/src/human.hexamer.logscore')
    lnc_SeqID, lnc_SeqList = list(lnc_seq_dict.keys()), list(lnc_seq_dict.values())
    lnc_HMMdict1 = ReadHmm(tmp_hmmer_out)
    return lnc_SeqID, lnc_SeqList, lnc_HMMdict1, logscore_dict

''' LCDS and Intrinsic features '''
def PartialFeature(seq, seqid, HMMdict1, logscore_dict):
    # HMMER index
    if seqid in HMMdict1:
        hmmfeature = HMMdict1[seqid][1] + "\t" + HMMdict1[seqid][2] + "\t" + HMMdict1[seqid][3]
    else:
        hmmfeature = "0\t0\t0"

    # Extract ORF
    seq = seq.strip()
    ORF, UTR5, UTR3, start, end = GetORF_UTR(seq)
    transcript_len = len(seq)

    # Coding sequence feature
    if len(seq) > 15:
        mlc_seq, ave_hexamer_score, mlc_start, mlc_end = MLC(seq, logscore_dict)
    else:
        mlc_seq, ave_hexamer_score, mlc_start, mlc_end = "", 0, 0, 0 

    if len(ORF) >= len(mlc_seq):
        tmp_seq = ORF
        ave_hexamer_score = HexamerScore2(tmp_seq, logscore_dict)
    else:
        tmp_seq = mlc_seq

    mlc_fea = str(len(tmp_seq)) + "\t" + str(len(tmp_seq) * 1.0 / transcript_len)

    ## EDP feature 
    if len(tmp_seq) < 6:
        EDP_fea = GetEDP_noORF()
    else:
        EDP_fea = GetEDP(tmp_seq, transcript_len)
    Kmer_EDP_fea = GetKmerEDP(tmp_seq)

    # Hexamer feature
    hex_score = str(ave_hexamer_score)

    # Fickett feature
    A_pos_fea = GetBasePositionScore(seq, 'A')
    C_pos_fea = GetBasePositionScore(seq, 'C')
    G_pos_fea = GetBasePositionScore(seq, 'G')
    T_pos_fea = GetBasePositionScore(seq, 'T')
    base_ratio = GetBaseRatio(seq)

    fickett_fea = "\t".join([str(A_pos_fea), str(C_pos_fea), str(G_pos_fea), str(T_pos_fea), str(base_ratio[0]), str(base_ratio[1]), str(base_ratio[2]), str(base_ratio[3])])

    # ORF integrity
    _, ORFintegrity, _ = ExtractORF(seq).longest_ORF() 
    ORFintegrity = str(ORFintegrity) 

    feature = "\t".join([EDP_fea, hmmfeature, Kmer_EDP_fea, hex_score, fickett_fea, mlc_fea, ORFintegrity])
    return feature

''' Add peptide features '''
def IntactFeature(lnc_para_tuple_list):
    lnc_feature = []
    for para_tuple in lnc_para_tuple_list:
        seqid = para_tuple[0]
        seq = para_tuple[1]
        HMMdict1 = para_tuple[2]
        logscore_dict = para_tuple[3]
        lnc_feature_tmp = PartialFeature(seq, seqid, HMMdict1, logscore_dict)
        lnc_feature_tmp = lnc_feature_tmp.split()
        lnc_feature_tmp = [float(fea) for fea in lnc_feature_tmp]
    
        # pep feature
        pep_fea = param(seq)
        pep_fea = [pep_fea[0], pep_fea[4]]

        lnc_feature_tmp = lnc_feature_tmp + pep_fea
        lnc_feature.append(lnc_feature_tmp)
    return lnc_feature


