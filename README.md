
##### TL;DR
This project was linked to a lecture about Shape Modeling. This project presents a method to deform meshs, based on high frequency details conservation and low frequency forms deformation, to keep a good integrity of the mesh during the edition.
<br/>
The editing method is real-time for small to medium meshes on a medium-class laptop computer.
<br/>

<p align="center">
<img style="height : 450px;" src="./results/multiresMethod.png">
<br/>Overview of the method - [Source]( http://graphics.uni-bielefeld.de )<br/>
</p>

##### Table of Contents  
[1. Multiresolution mesh editing](#Multiresolution)  <br/>
[1.1. Selecting the handles](#Selecting)  <br/>
[1.2. Removal of high-frequency details](#Removal)  <br/>
[1.3. Deforming the smooth mesh](#Deforming)  <br/>
[1.4. Transferring high-frequency details to the deformed surface](#Transferring)  <br/>
[1.5. Real-time performance](#Real)  <br/>
[2. Optional tasks](#Optional)  <br/>


<a name="Multiresolution"/>

#### 1. Multiresolution mesh editing

<a name="Selecting"/>

#### 1.1. Selecting the handles

The first step to deform a mesh is to select which part will move and which won't.
Therefore, you can use the GUI to select handle over the mesh.

__________________________________

<a name="Removal"/>

#### 1.2. Removal of high-frequency details

The next processing step is to remove, and store, high frequency details of the mesh.
To proceed, we create a copy of the mesh, that we heavily smooth, by minimizing the thing-plate energy. [Information on this energy](https://en.wikipedia.org/wiki/Thin_plate_energy_functional)

We will have to build and solve a system, to represent the minimization of this energy.

###### Documentation informations
We can use the following function : 
* __igl::invert_diag()__
* __igl::massmatrix()__
* __igl::cotmatrix()__
* __igl::adjacency_list()__
* __igl::per_vertex_normals()__

Informations on these functions were already provided for previous assignements. The others were not specified in the assignement.

###### Usage
The goal is to store the details of the original mesh, by storing the differences between the smoothed mesh and the original mesh.

First we smooth the mesh.
* We create the LML matrix (once for each mesh except if you change an option in the interface).
* We extract Aff and Afc from this matrix.
* We use a solver to obtain the smoothed positions of the free vertices.

We obtained the smoothed version of the mesh. Then we want to compute and store the difference between the original and smoothed mesh.

So we compute a local basis for each vertex, thanks to the Gram-Schmidt orthonormalization process applied to a 3-dimensions basis.
So, we will have 3 vectors : 
* First vector : we get the normal of each vertex of the mesh
* Second vector : we find the farest vertex-neighboor of the current vertex and we compute the projection on the plan of the vector between the current point and this neighboor
* Third vector :we get a third vector thanks to a cross product.

<p align="center">
<img style="height : 600px;" src="./results/Gram-Schmidt_orthonormalization_process.gif">
<br/>Gram Schmidt orthonormalization process (Public domain)<br/>
</p>

We then calculate the displacement vector between the original and the smoothed mesh, by an easy soustraction. We project this vector on the three axis on the basis created, by dot product.
We store the 3 factors for later use.

```C++

// == Save Old positions of vertices ==
  igl::slice(V, handle_vertices, 1, Handles_BeforeMove);
  igl::slice(V, free_vertices_indexes, 1, free_BeforeSmooth_Pos);

  (...)
  // == We compute the LML matrix (global variable)  ==

  //CREATION OF Lw
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, massMatrixType, M);
  igl::invert_diag(M, Minvert);

  LML = L * Minvert * L;

  // == Division of the Bilaplacian ==
  igl::slice(LML, free_vertices_indexes, free_vertices_indexes, Aff);
  igl::slice(LML, free_vertices_indexes, handle_vertices, Afc);

  solverCholesky.compute(Aff);

  //Creation of B (right side)
  RightSide = -1 * Afc * Handles_BeforeMove_Sparse;

  //Solving of the system
  Eigen::SparseMatrix<double> free_AfterSmooth_Pos_Sparse;

  free_AfterSmooth_Pos_Sparse = solverCholesky.solve(RightSide);

  (...)

  //Complete the base for each vertex
  Eigen::MatrixXd NormalsPerVertex;
  Eigen::MatrixXd T1;
  Eigen::MatrixXd T2;
  computeBasis(NormalsPerVertex, T1, T2, true); // true = creation mode of the basis

  // == Get the details  ==
  int index_currentFreeVertex = 0;

  for (int i = 0; i < free_vertices_indexes.rows(); i++)
  {
    index_currentFreeVertex = free_vertices_indexes[i];

    TMP_Diff = free_BeforeSmooth_Pos.row(i) - free_AfterSmooth_Pos.row(i);

    Difference.row(i)[0] = TMP_Diff.dot(NormalsPerVertex.row(index_currentFreeVertex));
    Difference.row(i)[1] = TMP_Diff.dot(T1.row(index_currentFreeVertex));
    Difference.row(i)[2] = TMP_Diff.dot(T2.row(index_currentFreeVertex));
  }

  (...)

  static void computeBasis(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, bool modeCreation)
{
  //Compute the normal of each vertex
  igl::per_vertex_normals(V, F, NormalsPerVertex);

  if (!CreatedAdjacency)
  {
    igl::adjacency_list(F, AdjacencyList);
    CreatedAdjacency = true;
  }

  //We create the basis
  for (int i = 0; i < free_vertices_indexes.rows(); i++)
  {
    int curFreePtsIndex = free_vertices_indexes[i];

    //Get the first vector
    Eigen::Matrix<double, 1, 3> x = NormalsPerVertex.row(curFreePtsIndex).normalized();
    int neighboorVertexID = -1;

    if (modeCreation)
    {
      //Take the "best" neighboor
      double maxDist = 0;
      int indexMaxDist = AdjacencyList[curFreePtsIndex][0]; //possible to blow up here if the vertex is alone

      for (int j = 0; j < AdjacencyList[curFreePtsIndex].size(); j++)
      {
        Eigen::Matrix<double, 1, 3> vectorToNeighboor = (V.row(AdjacencyList[curFreePtsIndex][j]) - V.row(curFreePtsIndex));
        Eigen::Matrix<double, 1, 3> projectionNeighboorOnX = (vectorToNeighboor).dot(x) * x; //X has a norm = 1
        double tempDist = (vectorToNeighboor - projectionNeighboorOnX).squaredNorm();

        if (maxDist < tempDist)
        {
          maxDist = tempDist;
          indexMaxDist = AdjacencyList[curFreePtsIndex][j];
        }
      }

      neighboorVertexID = indexMaxDist;
      //We store it for later use
      indiceNeighboor[curFreePtsIndex] = neighboorVertexID;

    }
    else
    {
      //We pick up consistenly
      neighboorVertexID = indiceNeighboor[curFreePtsIndex];
    }

    Eigen::Matrix<double, 1, 3> vectorToNeighboor = (V.row(neighboorVertexID) - V.row(curFreePtsIndex));
    Eigen::Matrix<double, 1, 3> projectionNeighboorOnX = (vectorToNeighboor.dot(x) / x.dot(x)) * x;
    Eigen::Matrix<double, 1, 3> y = (vectorToNeighboor - projectionNeighboorOnX).normalized();

    //Get the third vector
    Eigen::Matrix<double, 1, 3> z = x.cross(y).normalized(); // Or X Cross Y ?

    //Store the vectors
    NormalsPerVertex.row(curFreePtsIndex) = x.normalized();
    T1.row(curFreePtsIndex) = y.normalized();
    T2.row(curFreePtsIndex) = z.normalized();
  }

}

```

###### Result
<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="./results/woody_low-1.png">
  <img style="height : 300px;" src="./results/woody_low-2.png"><br/>
  <img style="height : 300px;" src="./results/woody-hi-1.png">
  <img style="height : 300px;" src="./results/woody-hi-2.png"><br/>
  <img style="width : 600px;" src="./results/bar-1.png">
  <img style="width : 600px;" src="./results/bar-2.png"><br/>
  <img style="height : 300px;" src="./results/plan-1.png">
  <img style="height : 300px;" src="./results/plan-2.png"><br/>
  <img style="height : 300px;" src="./results/camel-1.png">
  <img style="height : 300px;" src="./results/camel-2.png"><br/>
  <img style="height : 300px;" src="./results/cactus-1.png">
  <img style="height : 300px;" src="./results/cactus-2.png"><br/>
  <img style="height : 300px;" src="./results/cylinder-1.png">
  <img style="height : 300px;" src="./results/cylinder-2.png"><br/>
  <img style="height : 300px;" src="./results/hand-1.png">
  <img style="height : 300px;" src="./results/hand-2.png">
</p>
</div>

__________________________________


<a name="Deforming"/>

#### 1.3. Deforming the smooth mesh

Deforming the mesh is pretty straighforward, as it is similar to the previous smoothing operation, only with different constrained positions.

###### Usage

To deform the mesh, we apply the new positions of the handles moved by the user, and we solve again a system, to obtain the smoothed positions of the free vertices.

```C++

  (...)
  (Saving high frequency details)
  (...)

  //Move the handle
  igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);

  //Creation of B (right side)
  RightSide = -1 * Afc * handle_vertex_positionsSparse;

  //Solving of the system
  free_AfterSmooth_Pos_Sparse = solverCholesky.solve(RightSide);
  free_AfterSmooth_Pos = MatrixXd(free_AfterSmooth_Pos_Sparse);

  //Put the positions of free_AfterSmooth_Pos in view
  igl::slice_into(free_AfterSmooth_Pos, free_vertices_indexes, 1, V);

  (...)
```

###### Result
<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="./results/woody_low-3.png"> 
  <img style="height : 300px;" src="./results/woody-hi-3.png"><br/>
  <img style="height : 300px;" src="./results/bar-3.png">
  <img style="height : 300px;" src="./results/plan-3.png"><br/>
  <img style="height : 300px;" src="./results/cactus-3.png">
  <img style="height : 300px;" src="./results/camel-3.png"><br/>
  <img style="height : 300px;" src="./results/cylinder-3.png">
  <img style="height : 300px;" src="./results/hand-3.png"><br/>
</p>
</div>

__________________________________


<a name="Transferring"/>

#### 1.4. Transferring high-frequency details to the deformed surface

We then want to mesh to have all the details it had at the begining of the process. 
We will transfer high-frequency details previously stored to the deformed surface. As these details were encoded in local basis, we have to recompute these local basis in a consistent way.

###### Usage

So we want to put back the details, captured in each local basis of the smooth meshed,  in the smoothed deformed mesh.
<br/>
We recompute (same function) the local basis for each vertex of the mesh. Note, this time, we pick the same neighboor we choosed the first time and do not recalculate the farest one. This is the only difference.
The main reason is, that due to the deformation, the farest neighboor might have changed, and this may leads to inconsistencies.
<br/>
We then add to the actual position of each vertex of the deformed smoothed mesh, the displacement stored, projected in the local basis.

```C++

  (...)
  //Complete the base for each vertex
  computeBasis(NormalsPerVertex, T1, T2, false);

  // == Add back the details ==
  for (int i = 0; i < free_vertices_indexes.rows(); i++)
  {
    index_currentFreeVertex = free_vertices_indexes[i];

    // Calculate the addition (T1 = Y, T2 = Z, Normals = X), consistent with basis
    V.row(index_currentFreeVertex) =
        V.row(index_currentFreeVertex) +
        Difference.row(i)[0] * NormalsPerVertex.row(index_currentFreeVertex) +
        Difference.row(i)[1] * T1.row(index_currentFreeVertex) +
        Difference.row(i)[2] * T2.row(index_currentFreeVertex);
  }

  (...)

```

###### Result
<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="./results/woody_low-4.png">
  <img style="height : 300px;" src="./results/woody-hi-4.png"><br/>
  <img style="height : 300px;" src="./results/bar-4.png">
  <img style="height : 300px;" src="./results/plan-4.png"><br/>
  <img style="height : 300px;" src="./results/camel-4.png">
  <img style="height : 300px;" src="./results/cactus-4.png"><br/>
  <img style="height : 300px;" src="./results/cylinder-4.png">
  <img style="height : 300px;" src="./results/hand-4.png"><br/>

</p>
</div>

__________________________________


<a name="Real"/>

#### 1.5. Real-time performance

To acheive real-time perfomance (10+ fps) , we need the solving step of the system to be as fast as possible. An easy solution is to prefactor fixed part of the system, to avoid recomputation of static parts.
<br/>
We use the Cholesky factorization provided by Eigen.

###### Usage

To allow better performances, some more modifications are made : 
* Instead of specified function, only one "big" function solve is used. It saves some time for copies, and allow to reuse previous data structures. (same kind of optimization that we can find in videogames-industry : dirty and hard to maintain code, but efficient)
* Time-counting functions were added to follow were the time is spend during the computation. This does have an negative influence on the performance ! I decided to keep it to allow evaluation of the code.

Exemple of trace : 

```
Solve iteration (total 1 pass) time : 0.236447  // Total time spend to solve the current move
Solve timeTOsolveCompute : 0.043758             // Time before the call to .compute() of the solver, construting Aff, etc.
Solve compute time : 0.162786                   // Time to process the .compute() of the solver
Solve timeTOsolve1 : 0.000236                   // Time between the .compute() of the solver and the first .solve() of the system. Create some data structures
Solve old position time : 0.007828              // Time of the first solve of the system (old positions of handles) to get the smoothed old version of the mesh
Solve timeTOsolve2 : 0.004818                   // Time between the first and the second solve of the system, some data structure manipulation and save of the details.
Solve new position time : 0.013654              // Time of the second solve of the system, to get the new positins of the free vertices, with new positions of the handles
Solve timeTOEnd : 0.003366                      // Time to the end of the function, manipulating data structure
FrameRate : 4.22928                             // Framerate calculated from total elapsed time.
Note : this trace is from an old version of the application, before some optimisation.

```
* The Aff/Afc, LML calculations were moved into the onNewHandleID function, and are only calculated once during the selection of the handles. It allows 2 things : save approximately 50% of the calculation time (see previous trace, before this update) and make the deformations "reversibles", without artifact. (We can do a deformation and make it back without "blowing up" behaviors)
* Another optimization (called "Booster" in the interface) espace in time the new calculation. It's does not changed the calculation time but make the animation "look smoother". The gif of this report are not taken with this option activated. This is a visual improvement only.

```C++
  (...)
  if (showTime)
  {
    printTime();
  }

  if (showFrameRate)
  {
    printFrameRate();
  }
  (...)

  void printTime()
{
  std::cout << "Solve iteration (total 1 pass) time : " << timeTotal.getElapsedTime() << endl;
  std::cout << "Solve timeTOsolveCompute : " << timeTOsolveCompute.getElapsedTime() << endl;
  std::cout << "Solve compute time : " << timesolveCompute.getElapsedTime() << endl;
  std::cout << "Solve timeTOsolve1 : " << timeTOsolve1.getElapsedTime() << endl;
  std::cout << "Solve old position time : " << timesolve1.getElapsedTime() << endl;
  std::cout << "Solve timeTOsolve2 : " << timeTOsolve2.getElapsedTime() << endl;
  std::cout << "Solve new position time : " << timesolve2.getElapsedTime() << endl;
  std::cout << "Solve timeTOEnd : " << timeTOEnd.getElapsedTime() << endl;
}

void printFrameRate()
{
  std::cout << "FrameRate : " << 1 / timeTotal.getElapsedTime() << endl;
}
  (...)
```

###### Result
Note : timing are valid on a Intel Core i5 6300HQ 3.2 Ghz.

<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="./results/woody-low.gif">
</p>
</div>

```
woody-low
FrameRate : 1960.78
Solve iteration (total 1 pass) time : 0.00051
Solve timeTOsolveCompute : 1.9e-05
Solve compute time : 0
Solve timeTOsolve1 : 7.6e-05
Solve old position time : 6.7e-05
Solve timeTOsolve2 : 0.000189
Solve new position time : 5.3e-05
Solve timeTOEnd : 0.000104
```

<p align="center">
  <img style="height : 300px;" src="./results/woody-hi.gif">
</p>


```
woody-high
FrameRate : 315.06
Solve iteration (total 1 pass) time : 0.003174
Solve timeTOsolveCompute : 1.8e-05
Solve compute time : 0
Solve timeTOsolve1 : 4.6e-05
Solve old position time : 0.000801
Solve timeTOsolve2 : 0.000867
Solve new position time : 0.000799
Solve timeTOEnd : 0.000643

```

<p align="center">
  <img style="height : 300px;" src="./results/bar.gif">
</p>

```
bar
FrameRate : 90.1226
Solve iteration (total 1 pass) time : 0.011096
Solve timeTOsolveCompute : 4.7e-05
Solve compute time : 0
Solve timeTOsolve1 : 7.4e-05
Solve old position time : 0.00384
Solve timeTOsolve2 : 0.001847
Solve new position time : 0.003857
Solve timeTOEnd : 0.001431
```

<p align="center">
  <img style="height : 300px;" src="./results/plan.gif">
</p>

```
bumpy plan
FrameRate : 11.5867
Solve iteration (total 1 pass) time : 0.086306
Solve timeTOsolveCompute : 0.000315
Solve compute time : 0
Solve timeTOsolve1 : 0.00033
Solve old position time : 0.031813
Solve timeTOsolve2 : 0.011801
Solve new position time : 0.031935
Solve timeTOEnd : 0.010111
```

<p align="center">
  <img style="height : 300px;" src="./results/cactus.gif">
</p>

```
cactus
FrameRate : 86.1995
Solve iteration (total 1 pass) time : 0.011601
Solve timeTOsolveCompute : 0.000166
Solve compute time : 0
Solve timeTOsolve1 : 9.9e-05
Solve old position time : 0.006307
Solve timeTOsolve2 : 0.001841
Solve new position time : 0.002008
Solve timeTOEnd : 0.00118
```

<p align="center">
  <img style="height : 300px;" src="./results/camel.gif">
</p>

```
camel
FrameRate : 52.1349
Solve iteration (total 1 pass) time : 0.019181
Solve timeTOsolveCompute : 0.000103
Solve compute time : 1e-06
Solve timeTOsolve1 : 0.000306
Solve old position time : 0.006273
Solve timeTOsolve2 : 0.003633
Solve new position time : 0.006251
Solve timeTOEnd : 0.002614

```

<p align="center">
  <img style="height : 300px;" src="./results/cylinder.gif">
</p>

```
cylinder
FrameRate : 117.343
Solve iteration (total 1 pass) time : 0.008522
Solve timeTOsolveCompute : 2.9e-05
Solve compute time : 0
Solve timeTOsolve1 : 4.7e-05
Solve old position time : 0.002948
Solve timeTOsolve2 : 0.001443
Solve new position time : 0.002918
Solve timeTOEnd : 0.001137
```

<p align="center">
  <img style="height : 300px;" src="./results/hand.gif">
</p>

```
hand
FrameRate : 42.6185
Solve iteration (total 1 pass) time : 0.023464
Solve timeTOsolveCompute : 0.000141
Solve compute time : 1e-06
Solve timeTOsolve1 : 0.000183
Solve old position time : 0.007935
Solve timeTOsolve2 : 0.00409
Solve new position time : 0.007923
Solve timeTOEnd : 0.003191
```
__________________________________


Note that the code is not optimized by thoughtful laziness: this code is not made to be reused as is in other applications, and if this is the case, the code will be reviewed and optimized. Thanks for your understanding.
