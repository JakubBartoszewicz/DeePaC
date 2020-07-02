fetch 6W41

f = open("RBD_aas.tsv")
scores = [line for line in f]
f.close()
f = open("RBD_aas_nt.tsv")
scores_nt = [line for line in f]
f.close()

scores_cut = scores[14:209]
scores_nt_cut = scores_nt[14:209]

select RBD, model 6w41 and chain C
select AB, model 6w41 and (chain H or chain L)
select epitope, model 6w41 and chain C and (resi 369-372,374-386,389-390,392,427-430,515-517,519)
select not_epitope, model 6w41 and chain C and !(resi 369-372,374-386,389-390,392,427-430,515-517,519)


selection = "RBD"
gradient = "blue_white_red"
set transparency, 0.4, not_epitope
set transparency, 0.4, epitope
set transparency, 0.4, AB

names_cut = []
cmd.iterate(selection + " and n. ca and !solvent and alt a+\"\"", "names_cut.append(resn)")
nums_cut = []
cmd.iterate(selection + " and n. ca and !solvent and alt a+\"\"", "nums_cut.append(resi)")
    
for i in range(len(nums_cut)): cmd.alter(selection + " and resi " + nums_cut[i], "b=" + "0")
for i in range(len(nums_cut)): cmd.alter(selection + " and resi " + nums_cut[i], "b=" + scores_cut[i])
cmd.spectrum("b", gradient, selection=selection, minimum=-0.5, maximum=0.5)

scene F1, store, scores_rbd
png scores_out_rbd.png, dpi=300
scene F2, store, scores_epi
png scores_out_epi.png, dpi=300
scene F5, store, scores_rbd_side
png scores_side_out_rbd.png, dpi=300

scene F9, store, scores_ab_rbd
png scores_ab_rbd.png, dpi=300
scene F10, store, scores_ab_epi
png scores_ab_epi.png, dpi=300

for i in range(len(nums_cut)): cmd.alter(selection + " and resi " + nums_cut[i], "b=" + scores_nt_cut[i])
cmd.spectrum("b", gradient, selection=selection, minimum=-0.005, maximum=0.005)

scene F3, store, scores_nt_rbd
png scores_nt_rbd.png, dpi=300
scene F4, store, scores_nt_epi
png scores_nt_epi.png, dpi=300
scene F6, store, scores_rbd_side
png scores_side_nt_rbd.png, dpi=300

scene F11, store, scores_ab_nt_rbd
png scores_ab_nt_rbd.png, dpi=300
scene F12, store, scores_ab_epi
png scores_ab_nt_epi.png, dpi=300

