set I;  # colors
set J;  # available centroids

param d{j in J};            # distance between color i and centroid j
param X{i in I, j in J};    # cluster assignment matrix 
param k;                    # number of centroids to use

var y{j in J} binary;  # if centroid j and the corresponding cluster are chosen

minimize obj:
    sum{j in J} y[j] * d[j];

subject to c1{i in I}: sum{j in J}y[j] * X[i, j] = 1;       # each points is assigned to exactly 1 centroid
subject to c2: sum{j in J} y[j] = k;                        # exactly k centroids are used

solve;

display obj, {j in J: y[j] == 1};

end;