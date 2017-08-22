import numpy as np
    
def vc_dis_sym2(inpar):
    inst1, inst2 = inpar
    n1,n2,n3 = vc_dis_both(inst1, inst2)
    if n1==0:
        rst1_deform = 0.0
        rst1_nodeform = 0.0
    else:
        rst1_deform = n2/n1
        rst1_nodeform = n3/n1
    
    n1,n2,n3 = vc_dis_both(inst2, inst1)
    if n1==0:
        rst2_deform = 0.0
        rst2_nodeform = 0.0
    else: 
        rst2_deform = n2/n1
        rst2_nodeform = n3/n1
    
    return (np.mean([rst1_deform, rst2_deform]), np.mean([rst1_nodeform, rst2_nodeform]))

def vc_dis_paral(inpar):
    inst_ls, idx = inpar
    rst1 = np.ones(len(inst_ls))
    rst2 = np.ones(len(inst_ls))
    for nn in range(idx+1, len(inst_ls)):
        rst1[nn], rst2[nn] = vc_dis_sym2((inst_ls[idx], inst_ls[nn]))
        
    return (rst1, rst2)

    
def vc_dis_both(inst1, inst2):
    hh1,ww1,cc1 = inst1.shape
    assert(cc1 == inst2.shape[2])
    
    hh2 = inst2.shape[0]
    if hh1 > hh2:
        diff = hh1 - hh2
        diff_top = int(diff/2)
        diff_bottom = diff - diff_top
        inst2 = np.pad(inst2, ((diff_top, diff_bottom),(0,0),(0,0)), 'constant')
    elif hh1 < hh2:
        diff = hh2 - hh1
        diff_top = int(diff/2)
        inst2 = inst2[diff_top: diff_top+hh1,:,:]
        
    assert(hh1 == inst2.shape[0])
    
    ww2 = inst2.shape[1]
    if ww1 > ww2:
        diff = ww1 - ww2
        diff_top = int(diff/2)
        diff_bottom = diff - diff_top
        inst2 = np.pad(inst2, ((0,0),(diff_top, diff_bottom),(0,0)), 'constant')
    elif ww1 < ww2:
        diff = ww2 - ww1
        diff_top = int(diff/2)
        inst2 = inst2[:,diff_top: diff_top+ww1,:]
        
    assert(ww1 == inst2.shape[1])
    
    vc_dim = (hh1,ww1)
    dis_cnt_deform = 0
    dis_cnt_nodeform = 0
    where_f = np.where(inst2==1)
    for nn1, nn2, nn3 in zip(where_f[0], where_f[1], where_f[2]):
        hh_min = max(0,nn1-1)
        hh_max = min(hh1-1,nn1+1)
        ww_min = max(0,nn2-1)
        ww_max = min(ww1-1,nn2+1)
        
        if inst1[hh_min:hh_max+1, ww_min:ww_max+1, nn3].sum()==0:
            dis_cnt_deform += 1
        
        if inst1[nn1,nn2,nn3]==0:
            dis_cnt_nodeform += 1
            
    return (len(where_f[0]), dis_cnt_deform, dis_cnt_nodeform)