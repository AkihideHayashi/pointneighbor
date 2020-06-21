import pointneighbor as pn
from pointneighbor import PntFul, AdjSftSpc, VecSod, CelAdj

pe: PntFul
pn.cel_adj(pe, rc, num_div)
pn.cel_blg(cel_adj, pe)
pn.cel_num_div(cel_mat, rc)
pn.coo2_cel(cel_adj, blg, spc, ent)
pn.coo2_ful_pntsft(pe, rc)
pn.coo2_ful_simple(pe, rc)
# この辺のraw apiは纏めて違うところに置いておいた方がいいかも



pn.coo2_vec(adj, pos, cel)
pn.coo2_vec_sod(adj, pos, cel)
pn.lil2_vec(adj, pos, cel)
pn.lil2_vec_sod(adj, pos, cel)
pn.pnt_ful(cel, pbc, pos, ent)
# 分かりやすいのでそのまま採用


adj, vec_sod = pn.cutoff_coo2(adj, pos, cel, rc)
# この関数はカットオフ処理を含むが、名前がそれを表していないし、それならvec_sodを使い回した方が良い。

pn.transformation_mask_coo_to_lil(adj_coo)  # transformation_mask_coo_to_lil ?
pn.transform_tensor_coo_to_lil(coo, mask, dummy)  # coo tensorをlil tensorに変換だが、adjをlilにするのと勘違いしがち
# transform_tensor_coo_to_lilとかにした方がいいかも

pn.coo_to_lil(adj_coo)  # その機能はこっち. これこそ coo_to_lilって感じ。もしくはadj_coo_to_lil?
pn.lil2_adj_sft_spc_vec_sod(adj_coo, vec_sod_coo)  # この関数はそもそもtransformに使い方の解説のためだけに存在する。上に合わせて名前を変えて良い。
pn.vec_sod(adj, pos, cel)  # coo2_vec_sodとlil2_vec_sodを内部で分岐するが、これ要る？

pn.get_lil2_j_s(adj)  # schnetpackで便利
pn.get_n_i_j(adj)  # せめてget_coo_n_i_j にしないと上と同じようにならない
pn.get_n_i_j_s(adj)  # というか、getいる？
# この辺は名前を変えた方がいいかも
