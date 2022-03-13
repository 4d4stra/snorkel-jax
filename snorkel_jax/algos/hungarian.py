import jax.numpy as jnp

def _mark_elems_(res_mat,mat_assigned,marked_rows,marked_cols,marked_rows_i):
    count_row_init=jnp.sum(marked_rows)
    marked_cols_i=jnp.sum(res_mat[marked_rows_i]==0,axis=0)>0
    if jnp.sum(marked_cols_i)>0:
        #mark rows
        marked_rows_i=jnp.max(mat_assigned[:,marked_cols_i],axis=1)
    else:
        marked_rows_i=jnp.array([False]*len(res_mat))
    #merging
    marked_cols=jnp.maximum(marked_cols,marked_cols_i)
    marked_rows=jnp.maximum(marked_rows,marked_rows_i)
    #recurse
    if jnp.sum(marked_rows)>count_row_init:#new rows added
        marked_cols,marked_rows=_mark_elems_(res_mat,mat_assigned,marked_rows,marked_cols,marked_rows_i)

    return marked_cols,marked_rows

#get current assigned elements
def _get_assignments_(res_mat):
    new_assignments=1
    mat_assigned=jnp.zeros(res_mat.shape)==1
    while new_assignments>0:
        res_mat_i=res_mat+jnp.amax(mat_assigned,axis=0)+jnp.amax(mat_assigned,axis=1).reshape((len(res_mat),1))
        if jnp.sum(jnp.sum(res_mat_i==0,axis=1)==1)>0:
            res_mat_ii=res_mat_i+jnp.maximum(jnp.sum(res_mat_i==0,axis=1)-1,0).reshape((len(res_mat),1))
            mat_assigned_i=(res_mat_ii==0) \
                    & (jnp.cumsum(res_mat_ii==0,axis=1)==1) \
                    & (jnp.cumsum(res_mat_ii==0,axis=0)==1)
        else:
            mat_assigned_i=(res_mat_i==0) \
                    & (jnp.cumsum(res_mat_i==0,axis=1)==1) \
                    & (jnp.cumsum(res_mat_i==0,axis=0)==1)
        new_assignments=jnp.sum(mat_assigned_i)
        mat_assigned=jnp.maximum(mat_assigned,mat_assigned_i)
    return mat_assigned

def solve(res_mat):
    #Subtract row mins
    res_mat=res_mat-jnp.min(res_mat,axis=1).reshape((len(res_mat),1))

    #Subtract col mins
    res_mat=res_mat-jnp.min(res_mat,axis=0)

    mat_assigned=_get_assignments_(res_mat)
    n_assigned=jnp.sum(mat_assigned)
    while n_assigned<len(res_mat):
        marked_rows=~jnp.amax(mat_assigned,axis=1)
        marked_cols,marked_rows=_mark_elems_(res_mat,mat_assigned,marked_rows,jnp.array([False]*len(res_mat)),marked_rows)

        #find the min of the unmarked subset
        leftover_mask=jnp.ones(res_mat.shape)==1
        leftover_mask=leftover_mask.at[~marked_rows].set(False)
        leftover_mask=leftover_mask.at[:,marked_cols].set(False)
        leftover_min=jnp.amin(res_mat[leftover_mask])
        
        #adjusting matrix
        res_mat=res_mat.at[~marked_rows].set(res_mat[~marked_rows]+leftover_min)
        res_mat=res_mat.at[:,marked_cols].set(res_mat[:,marked_cols]+leftover_min)
        res_mat=res_mat-leftover_min
        
        #reassigning
        mat_assigned=_get_assignments_(res_mat)
        n_assigned=jnp.sum(mat_assigned)
    return mat_assigned