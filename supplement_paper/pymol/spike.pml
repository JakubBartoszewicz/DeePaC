fetch 6VSB

f = open("RBD_aas.tsv")
scores = [line for line in f]
f.close()
f = open("RBD_aas_nt.tsv")
scores_nt = [line for line in f]
f.close()

select S_RBD, model 6vsb and resi 319-541
select S_epitope, model 6vsb and (resi 369-372,374-386,389-390,392,427-430,515-517,519)
select S_not_epitope, model 6vsb and resi 319-541 and !(resi 369-372,374-386,389-390,392,427-430,515-517,519)
select S_not_RBD, model 6vsb and !(resi 319-541)

selection = "S_RBD"
gradient = "blue_white_red"
nums = [str(i) for i in range(319,542)]

for i in range(len(nums)): cmd.alter(selection + " and resi " + nums[i], "b=" + "0")
for i in range(len(scores)): cmd.alter(selection + " and resi " + nums[i], "b=" + scores[i])
cmd.spectrum("b", gradient, selection=selection, minimum=-0.5, maximum=0.5)
set transparency, 0.5, S_not_RBD
set transparency, 0, S_RBD
set transparency, 0, S_epitope


scene F8, store, scores_s_top
png scores_s_out_top.png, dpi=300
turn x, -90
turn y, 60
scene F9, store, scores_s_right
png scores_s_out_right.png, dpi=300
turn y, 180
scene F7, store, scores_s_left
png scores_s_out_left.png, dpi=300

for i in range(len(scores_nt)): cmd.alter(selection + " and resi " + nums[i], "b=" + scores_nt[i])
cmd.spectrum("b", gradient, selection=selection, minimum=-0.005, maximum=0.005)

png scores_s_nt_top.png, dpi=300
png scores_s_nt_right.png, dpi=300
png scores_s_nt_left.png, dpi=300

names_epitope = names[50:54] + names[55:68] + names[70:72] + [names[73]] + names[108:112] + names[196:199] + [names[200]]
scores_epitope = scores[50:54] + scores[55:68] + scores[70:72] + [scores[73]] + scores[108:112] + scores[196:199] + [scores[200]]
scores_nt_epitope = scores_nt[50:54] + scores_nt[55:68] + scores_nt[70:72] + [scores_nt[73]] + scores_nt[108:112] + scores_nt[196:199] + [scores_nt[200]]
