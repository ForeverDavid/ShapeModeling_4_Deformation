#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/Timer.h>

#include "Lasso.h"
#include "Colors.h"

#include <igl/cotmatrix.h>
#include <Eigen/Geometry>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/adjacency_list.h>

//#define DEBUG

/*** insert any necessary libigl headers here ***/
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>

//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;
using namespace Eigen;
using Viewer = igl::viewer::Viewer;

igl::viewer::Viewer viewer;

bool CreatedAdjacency = false;
std::vector<std::vector<int>> AdjacencyList; //Or double ?

bool CreatedIndice = false;
Eigen::VectorXi indiceNeighboor;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0, 3);
int NbLineToDisplay = 10;

Eigen::MatrixXd Difference;

int status = 0;

static void computeBasis(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, bool modeCreation);
static void computeDetails(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, Eigen::MatrixXd &BeforePos, Eigen::MatrixXd &AfterPos, Eigen::MatrixXd &Difference);
static void addDetails(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, Eigen::MatrixXd &Difference);

//mouse interaction
enum MouseMode
{
  SELECT,
  TRANSLATE,
  ROTATE,
  NONE
};
MouseMode mouse_mode = NONE;
bool doit = false;
int down_mouse_x = -1, down_mouse_y = -1;

//for selecting vertices
std::unique_ptr<Lasso> lasso;
//list of currently selected vertices
Eigen::VectorXi selected_v(0, 1);

//for saving constrained vertices
//vertex-to-handle index, #V x1 (-1 if vertex is free)
Eigen::VectorXi handle_id(0, 1);
//list of all vertices belonging to handles, #HV x1
Eigen::VectorXi handle_vertices(0, 1);
//centroids of handle regions, #H x1
Eigen::MatrixXd handle_centroids(0, 3);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0, 3);

//ADDED
//list of all vertices belonging to handles, #V-HV x1
Eigen::VectorXi free_vertices_indexes(0, 1);
//updated positions of handle vertices, #V-HV x3
Eigen::MatrixXd non_handle_vertex_positions(0, 3);

igl::MassMatrixType massMatrixType = igl::MASSMATRIX_TYPE_DEFAULT; // MASSMATRIX_TYPE_VORONOI ?

//index of handle being moved
int moving_handle = -1;
//rotation and translation for the handle being moved
Eigen::Vector3f translation(0, 0, 0);
Eigen::Vector4f rotation(0, 0, 0, 1.);

//per vertex color array, #V x3
Eigen::MatrixXd vertex_colors;

//function declarations (see below for implementation)
bool solve(igl::viewer::Viewer &viewer);
void get_new_handle_locations();
Eigen::Vector3f computeTranslation(igl::viewer::Viewer &viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
Eigen::Vector4f computeRotation(igl::viewer::Viewer &viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
void compute_handle_centroids();
Eigen::MatrixXd readMatrix(const char *filename);

bool callback_mouse_down(igl::viewer::Viewer &viewer, int button, int modifier);
bool callback_mouse_move(igl::viewer::Viewer &viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(igl::viewer::Viewer &viewer, int button, int modifier);
bool callback_pre_draw(igl::viewer::Viewer &viewer);
bool callback_key_down(igl::viewer::Viewer &viewer, unsigned char key, int modifiers);
void onNewHandleID();
void applySelection();
void printTime();
void printFrameRate();

/*
- On solve le système une première fois avec les points contraints ayant pour coordonnées leurs coordonnées initiales (dans V)
- Tu calcul la différence entre ce smoothed mesh et l'initiale mesh (différence de 0 pour les points contraints, d'autres chose pour les points non contraints)
- On solve le système une deuxième fois, avev les points contraints ayant oour coordonnées leurs nouvelles coordonnés (dans handle_vertex_positions)
- Tu rajoutes les valeurs sauvées en 2)
*/
Eigen::SparseMatrix<double> LML;
Eigen::SparseMatrix<double> Aff;
Eigen::SparseMatrix<double> Afc;
Eigen::SimplicialCholesky<SparseMatrix<double>, Eigen::RowMajor> solverCholesky;
Eigen::SimplicialCholesky<Eigen::MatrixXd, Eigen::RowMajor> solverCholesky_Dense;

static void computeLML()
{
#ifdef DEBUG
  std::cout << " > Dimensions of V " << V.rows() << " rows by " << V.cols() << " cols." << endl;
  std::cout << " > Dimensions of F " << F.rows() << " rows by " << F.cols() << " cols." << endl;
#endif

  //CREATION OF Lw
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V, F, L); //Calculate cotan laplacian of the Shape (whole)

#ifdef DEBUG
  std::cout << "Cotmatrix Overview : " << endl;
  Eigen::MatrixXd printableL = MatrixXd(L);
  for (int i = 0; i < printableL.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableL.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableL.row(0).size(); j++)
      {
        std::cout << printableL.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of cotmatrix Overview" << endl;
#endif

  // CREATION OF M
  Eigen::SparseMatrix<double> M;
  Eigen::SparseMatrix<double> Minvert;
  //computeMInverted(M);
  igl::massmatrix(V, F, massMatrixType, M);
  igl::invert_diag(M, Minvert);

#ifdef DEBUG
  std::cout << " > Dimensions of M " << M.rows() << " rows by " << M.cols() << " cols." << endl;
  std::cout << " > Dimensions of L " << L.rows() << " rows by " << L.cols() << " cols." << endl;

  std::cout << " M Overview : " << endl;
  Eigen::MatrixXd printableM = MatrixXd(M);
  for (int i = 0; i < printableM.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableM.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableM.row(0).size(); j++)
      {
        std::cout << printableM.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of M Overview" << endl;
#endif

  //CREATION OF the "big matrix"
  LML = L * Minvert * L;

#ifdef DEBUG
  std::cout << " > Dimensions of LML " << LML.rows() << " rows by " << LML.cols() << " cols." << endl;

  std::cout << "LML Overview : " << endl;
  Eigen::MatrixXd printableLML = MatrixXd(LML);
  for (int i = 0; i < printableLML.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableLML.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableLML.row(0).size(); j++)
      {
        std::cout << printableLML.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printableLML Overview" << endl;
#endif
}

static void getVertexPositionsBeforeMove(SparseMatrix<double> &Handles_BeforeMove_Sparse, SparseMatrix<double> &free_BeforeSmooth_Pos_Sparse)
{
  //Get old positions
  Eigen::MatrixXd Handles_BeforeMove;
  Eigen::MatrixXd free_BeforeSmooth_Pos;

  //Resize
  Handles_BeforeMove.resize(handle_vertices.rows(), 3);
  free_BeforeSmooth_Pos.resize(free_vertices_indexes.rows(), 3);

  //Get only the interseting lines of V +   //Get Sparse version (dumb but necessary for solving)
  igl::slice(V, handle_vertices, 1, Handles_BeforeMove);
  igl::slice(V, free_vertices_indexes, 1, free_BeforeSmooth_Pos);

  //Reput in sparse view
  Handles_BeforeMove_Sparse = Handles_BeforeMove.sparseView();
  free_BeforeSmooth_Pos_Sparse = free_BeforeSmooth_Pos.sparseView();

#ifdef DEBUG
  std::cout << "Handles_BeforeMove_Sparse Overview : " << endl;
  Eigen::MatrixXd printableHandles_BeforeMove_Sparse = MatrixXd(Handles_BeforeMove_Sparse);
  for (int i = 0; i < printableHandles_BeforeMove_Sparse.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableHandles_BeforeMove_Sparse.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableHandles_BeforeMove_Sparse.row(0).size(); j++)
      {
        std::cout << printableHandles_BeforeMove_Sparse.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printableHandles_BeforeMove_Sparse Overview" << endl;

  std::cout << "free_BeforeSmooth_Pos Overview : " << endl;
  Eigen::MatrixXd printablefree_BeforeSmooth_Pos = MatrixXd(free_BeforeSmooth_Pos);
  for (int i = 0; i < printablefree_BeforeSmooth_Pos.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printablefree_BeforeSmooth_Pos.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printablefree_BeforeSmooth_Pos.row(0).size(); j++)
      {
        std::cout << printablefree_BeforeSmooth_Pos.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printablefree_BeforeSmooth_Pos Overview" << endl;

  std::cout << " > Dimensions of Handles_BeforeMove_Sparse (lines of V corresponding to handles) " << Handles_BeforeMove_Sparse.rows() << " rows by " << Handles_BeforeMove_Sparse.cols() << " cols." << endl;
  std::cout << " > Dimensions of free_BeforeSmooth_Pos_Sparse (lines of V corresponding to NOT handles) " << free_BeforeSmooth_Pos_Sparse.rows() << " rows by " << free_BeforeSmooth_Pos_Sparse.cols() << " cols." << endl;
#endif
}

static void getVertexPositionsBeforeMove(Eigen::MatrixXd &Handles_BeforeMove, Eigen::MatrixXd &free_BeforeSmooth_Pos)
{
  //Resize
  Handles_BeforeMove.resize(handle_vertices.rows(), 3);
  free_BeforeSmooth_Pos.resize(free_vertices_indexes.rows(), 3);

  //Get only the interseting lines of V +   //Get Sparse version (dumb but necessary for solving)
  igl::slice(V, handle_vertices, 1, Handles_BeforeMove);
  igl::slice(V, free_vertices_indexes, 1, free_BeforeSmooth_Pos);
}

static void solveSystem(SparseMatrix<double> &Right, Eigen::MatrixXd &free_AfterSmooth_Pos)
{
  //Solving of the system
  Eigen::SparseMatrix<double> notConstrainedPointsPosition;
  Eigen::SparseMatrix<double> free_AfterSmooth_Pos_Sparse;

  free_AfterSmooth_Pos_Sparse = solverCholesky.solve(Right);
  //free_AfterSmooth_Pos_Sparse = solverCholeskyDense.solve(Right);

  free_AfterSmooth_Pos = MatrixXd(free_AfterSmooth_Pos_Sparse);
  assert(free_AfterSmooth_Pos.cols() == 3);

#ifdef DEBUG
  std::cout << " > Dimensions of Left side " << Left.rows() << " rows by " << Left.cols() << " cols." << endl;
  std::cout << " > Dimensions of Right side " << Right.rows() << " rows by " << Right.cols() << " cols." << endl;
  std::cout << " > Dimensions of free_AfterSmooth_Pos_Sparse side " << free_AfterSmooth_Pos_Sparse.rows() << " rows by " << free_AfterSmooth_Pos_Sparse.cols() << " cols." << endl;
#endif

#ifdef DEBUG
  std::cout << " free_AfterSmooth_Pos Overview : " << endl;
  for (int i = 0; i < free_AfterSmooth_Pos.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > free_AfterSmooth_Pos.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < free_AfterSmooth_Pos.row(0).size(); j++)
      {
        std::cout << free_AfterSmooth_Pos.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of free_AfterSmooth_Pos Overview" << endl;
#endif
}

static void computeBasis(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, bool modeCreation)
{
  //Compute the normal of each vertex
  igl::per_vertex_normals(V, F, NormalsPerVertex);

  if (!CreatedAdjacency)
  {
    igl::adjacency_list(F, AdjacencyList);
    CreatedAdjacency = true;
  }

  //NOTE : No need to compute it each time. Boolean only for the first time
  if (!modeCreation)
  {
    //We pick up consistenly
    assert(CreatedAdjacency);
    assert(CreatedIndice);
    assert(indiceNeighboor.rows() == V.rows());
  }
  else if (modeCreation)
  {
    indiceNeighboor.resize(V.rows());
    indiceNeighboor.setZero();
  }

  T1.resize(V.rows(), 3);
  T2.resize(V.rows(), 3);
  T1.setZero();
  T2.setZero();

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

#ifdef DEBUG
      std::cout << "Choosed neighboor : " << indexMaxDist << endl;
#endif
      //OLD : For now, just always take the first one :
      //OLD : neighboorVertexID = AdjacencyList[i][1];
    }
    else
    {
      //We pick up consistenly
      neighboorVertexID = indiceNeighboor[curFreePtsIndex];
    }

    assert(neighboorVertexID >= 0);

    Eigen::Matrix<double, 1, 3> vectorToNeighboor = (V.row(neighboorVertexID) - V.row(curFreePtsIndex));
    Eigen::Matrix<double, 1, 3> projectionNeighboorOnX = (vectorToNeighboor.dot(x) / x.dot(x)) * x;
    Eigen::Matrix<double, 1, 3> y = (vectorToNeighboor - projectionNeighboorOnX).normalized();

    //Get the third vector
    Eigen::Matrix<double, 1, 3> z = x.cross(y).normalized(); // Or X Cross Y ?

    //Store the vectors
    NormalsPerVertex.row(curFreePtsIndex) = x.normalized();
    T1.row(curFreePtsIndex) = y.normalized();
    T2.row(curFreePtsIndex) = z.normalized();

#ifdef DEBUG
    std::cout << " Vertice N° " << i << endl;
    std::cout << " Vector chosen for Normal (X) \t" << NormalsPerVertex.row(i) << endl;
    std::cout << " Vector chosen for T1 (Y) \t" << T1.row(i) << endl;
    std::cout << " Vector chosen for T2 (Z) \t" << T2.row(i) << endl;
#endif
  }

  if (modeCreation)
  {
    CreatedIndice = true;
  }
}

static void computeDetails(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, Eigen::MatrixXd &BeforePos, Eigen::MatrixXd &AfterPos, Eigen::MatrixXd &Difference)
{
  assert(NormalsPerVertex.rows() == V.rows());
  assert(T1.rows() == V.rows());
  assert(T2.rows() == V.rows());
  assert(BeforePos.rows() == free_vertices_indexes.rows());
  assert(AfterPos.rows() == free_vertices_indexes.rows());

  //Calcul in global basis
  Difference.resize(free_vertices_indexes.rows(), 3);
  Difference = BeforePos - AfterPos; //invert ?

  Eigen::VectorXd TMP_Diff(1, 3);
  Eigen::VectorXd TMP_X(3, 1);
  Eigen::VectorXd TMP_Y(3, 1);
  Eigen::VectorXd TMP_Z(3, 1);
  int index_currentFreeVertex = 0;

  for (int i = 0; i < free_vertices_indexes.rows(); i++)
  {
    index_currentFreeVertex = free_vertices_indexes[i];

    TMP_Diff = Difference.row(i); //Note : difference has the size of the list of free vertices

    TMP_Y = T1.row(index_currentFreeVertex).transpose();
    TMP_Z = T2.row(index_currentFreeVertex).transpose();
    TMP_X = NormalsPerVertex.row(index_currentFreeVertex).transpose();

    // Calculate the project (T1 = Y, T2 = Z, Normals = X), consistent with basis
    //Store it
    Difference.row(i)[0] = TMP_Diff.dot(NormalsPerVertex.row(index_currentFreeVertex));
    Difference.row(i)[1] = TMP_Diff.dot(T1.row(index_currentFreeVertex));
    Difference.row(i)[2] = TMP_Diff.dot(T2.row(index_currentFreeVertex));
  }
}

static void addDetails(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, Eigen::MatrixXd &Difference)
{
  assert(NormalsPerVertex.rows() == V.rows());
  assert(T1.rows() == V.rows());
  assert(T2.rows() == V.rows());

  assert(Difference.rows() == free_vertices_indexes.rows());

  //Calcul in global basis
  Eigen::VectorXd TMP_Diff(3, 1);
  Eigen::VectorXd TMP_X(3, 1);
  Eigen::VectorXd TMP_Y(3, 1);
  Eigen::VectorXd TMP_Z(3, 1);
  int index_currentFreeVertex = 0;

  for (int i = 0; i < free_vertices_indexes.rows(); i++)
  {
    index_currentFreeVertex = free_vertices_indexes[i];

    TMP_Y = T1.row(index_currentFreeVertex).transpose();
    TMP_Z = T2.row(index_currentFreeVertex).transpose();
    TMP_X = NormalsPerVertex.row(index_currentFreeVertex).transpose();

    // Calculate the addition (T1 = Y, T2 = Z, Normals = X), consistent with basis
    Eigen::VectorXd currentPos = V.row(index_currentFreeVertex).transpose();
    V.row(index_currentFreeVertex) = (V.row(index_currentFreeVertex) + Difference.row(i)[0] * NormalsPerVertex.row(index_currentFreeVertex) + Difference.row(i)[1] * T1.row(index_currentFreeVertex) + Difference.row(i)[2] * T2.row(index_currentFreeVertex));

    //AD DEBUG HERE
  }
}

static void saveHighFrequency()
{
  //We compute the LML matrix (global variable)
  computeLML();

  //Get old positions
  Eigen::SparseMatrix<double> Handles_BeforeMove_Sparse;    //The previous positions of Handles (before moving)
  Eigen::SparseMatrix<double> free_BeforeSmooth_Pos_Sparse; //The previous positions of not-Handles (before moving/smoothing)

  getVertexPositionsBeforeMove(Handles_BeforeMove_Sparse, free_BeforeSmooth_Pos_Sparse);

  //Division of the Bilaplacian
  igl::slice(LML, free_vertices_indexes, free_vertices_indexes, Aff);
  igl::slice(LML, free_vertices_indexes, handle_vertices, Afc);

  solverCholesky.compute(Aff);
  //solverCholesky_Dense.compute(Aff);

#ifdef DEBUG
  std::cout << " > Dimensions of Aff " << Aff.rows() << " rows by " << Aff.cols() << " cols." << endl;
  std::cout << " > Dimensions of Afc " << Afc.rows() << " rows by " << Afc.cols() << " cols." << endl;

  std::cout << "Aff Overview : " << endl;
  Eigen::MatrixXd printableAff = MatrixXd(Aff);
  for (int i = 0; i < printableAff.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableAff.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableAff.row(0).size(); j++)
      {
        std::cout << printableAff.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of Aff Overview" << endl;

  std::cout << "Afc Overview : " << endl;
  Eigen::MatrixXd printableAfc = MatrixXd(Afc);
  for (int i = 0; i < printableAfc.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableAfc.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableAfc.row(0).size(); j++)
      {
        std::cout << printableAfc.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of Afc Overview" << endl;
#endif

  //CREATION OF A (left side)
  //It's just Aff

  //Creation of B (right side)
  Eigen::SparseMatrix<double> RightSide;
  RightSide = -1 * Afc * Handles_BeforeMove_Sparse;

#ifdef DEBUG
  std::cout << "RightSide Overview : " << endl;
  Eigen::MatrixXd printableRightSide = MatrixXd(RightSide);
  for (int i = 0; i < printableRightSide.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableRightSide.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableRightSide.row(0).size(); j++)
      {
        std::cout << printableRightSide.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printableRightSide Overview" << endl;
#endif

  //Solving of the system
  Eigen::MatrixXd free_AfterSmooth_Pos;
  solveSystem(RightSide, free_AfterSmooth_Pos);

  //Put the positions of free_AfterSmooth_Pos in view
  igl::slice_into(free_AfterSmooth_Pos, free_vertices_indexes, 1, V);
  // SURE TO DO IT BEFORE CALCULATION OF BASIS ? YES : we need to get the old psoitions thanks to the new

  //==> GET HIGH FREQUENCY DETAILS
  //Complete the base for each vertex
  Eigen::MatrixXd NormalsPerVertex;
  Eigen::MatrixXd T1;
  Eigen::MatrixXd T2;
  computeBasis(NormalsPerVertex, T1, T2, true); // true = creation mode of the basis

  //Get the details
  Eigen::MatrixXd free_BeforeSmooth_Pos = Eigen::MatrixXd(free_BeforeSmooth_Pos_Sparse); //The previous positions of not-Handles (before moving/smoothing)
  computeDetails(NormalsPerVertex, T1, T2, free_BeforeSmooth_Pos, free_AfterSmooth_Pos, Difference);
}

static void deformMesh()
{
  //Move the handle
  igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);

  //Solve the new system

  //Get the new positions of the handles
  Eigen::SparseMatrix<double> handle_vertex_positionsSparse;
  handle_vertex_positionsSparse = handle_vertex_positions.sparseView();

  //CREATION OF A (left side)
  //It's just Aff

  //Creation of B (right side)
  Eigen::SparseMatrix<double> RightSide;
  RightSide = -1 * Afc * handle_vertex_positionsSparse;

#ifdef DEBUG
  std::cout << "RightSide Overview : " << endl;
  Eigen::MatrixXd printableRightSide = MatrixXd(RightSide);
  for (int i = 0; i < printableRightSide.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableRightSide.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableRightSide.row(0).size(); j++)
      {
        std::cout << printableRightSide.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printableRightSide Overview" << endl;
#endif

  //Solving of the system
  Eigen::MatrixXd free_AfterSmooth_Pos;
  solveSystem(RightSide, free_AfterSmooth_Pos);

  //Put the positions of free_AfterSmooth_Pos in view
  igl::slice_into(free_AfterSmooth_Pos, free_vertices_indexes, 1, V);
}

static void addHighFrequency()
{
  //==> GET HIGH FREQUENCY DETAILS

  //Complete the base for each vertex
  Eigen::MatrixXd NormalsPerVertex;
  Eigen::MatrixXd T1;
  Eigen::MatrixXd T2;
  computeBasis(NormalsPerVertex, T1, T2, false);

  //Add back the details
  addDetails(NormalsPerVertex, T1, T2, Difference);
}

bool showTime = false;
bool showFrameRate = false;
bool LMLDependsonPositions = false;
igl::Timer timeTotal;
igl::Timer timeTOsolveCompute;
igl::Timer timesolveCompute;
igl::Timer timeTOsolve1;
igl::Timer timesolve1;
igl::Timer timeTOsolve2;
igl::Timer timesolve2;
igl::Timer timeTOEnd;

bool solve(igl::viewer::Viewer &viewer)
{
  timeTotal.start();
  timeTOsolveCompute.start();

  // == Save Old positions of vertices ==
  Eigen::SparseMatrix<double> Handles_BeforeMove_Sparse;
  Eigen::MatrixXd Handles_BeforeMove;
  Eigen::MatrixXd free_BeforeSmooth_Pos;

  igl::slice(V, handle_vertices, 1, Handles_BeforeMove);
  igl::slice(V, free_vertices_indexes, 1, free_BeforeSmooth_Pos);

  Handles_BeforeMove_Sparse = Handles_BeforeMove.sparseView();

  // == We compute the LML matrix (global variable)  ==

  if (LMLDependsonPositions)
  {
    //CREATION OF Lw
    Eigen::SparseMatrix<double> L;
    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> Minvert;

    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, massMatrixType, M);
    igl::invert_diag(M, Minvert);

    LML = L * Minvert * L;

    // == Division of the Bilaplacian ==
    igl::slice(LML, free_vertices_indexes, free_vertices_indexes, Aff);
    igl::slice(LML, free_vertices_indexes, handle_vertices, Afc);

    //Time managing
    timeTOsolveCompute.stop();
    timesolveCompute.start();
    timesolveCompute.stop();
    timeTOsolve1.start();
  }
  else
  {
    timeTOsolveCompute.stop();
    timesolveCompute.start();
    solverCholesky.compute(Aff);
    timesolveCompute.stop();
    timeTOsolve1.start();
  }

  //Creation of B (right side)
  Eigen::SparseMatrix<double> RightSide;
  RightSide = -1 * Afc * Handles_BeforeMove_Sparse;

  //Solving of the system
  Eigen::MatrixXd free_AfterSmooth_Pos;
  Eigen::SparseMatrix<double> free_AfterSmooth_Pos_Sparse;

  timeTOsolve1.stop();
  timesolve1.start();
  free_AfterSmooth_Pos_Sparse = solverCholesky.solve(RightSide);
  free_AfterSmooth_Pos = MatrixXd(free_AfterSmooth_Pos_Sparse);
  timesolve1.stop();
  timeTOsolve2.start();

  igl::slice_into(free_AfterSmooth_Pos, free_vertices_indexes, 1, V);

  //Complete the base for each vertex
  Eigen::MatrixXd NormalsPerVertex;
  Eigen::MatrixXd T1;
  Eigen::MatrixXd T2;
  computeBasis(NormalsPerVertex, T1, T2, true); // true = creation mode of the basis

  // == Get the details  ==
  //computeDetails(NormalsPerVertex, T1, T2, free_BeforeSmooth_Pos, free_AfterSmooth_Pos, Difference);

  Eigen::VectorXd TMP_Diff(1, 3);
  int index_currentFreeVertex = 0;
  Difference.resize(free_vertices_indexes.rows(), 3);

  for (int i = 0; i < free_vertices_indexes.rows(); i++)
  {
    index_currentFreeVertex = free_vertices_indexes[i];

    TMP_Diff = free_BeforeSmooth_Pos.row(i) - free_AfterSmooth_Pos.row(i);

    Difference.row(i)[0] = TMP_Diff.dot(NormalsPerVertex.row(index_currentFreeVertex));
    Difference.row(i)[1] = TMP_Diff.dot(T1.row(index_currentFreeVertex));
    Difference.row(i)[2] = TMP_Diff.dot(T2.row(index_currentFreeVertex));
  }

  //Move the handle
  igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);

  //Get the new positions of the handles
  Eigen::SparseMatrix<double> handle_vertex_positionsSparse;
  handle_vertex_positionsSparse = handle_vertex_positions.sparseView();

  //Creation of B (right side)
  RightSide = -1 * Afc * handle_vertex_positionsSparse;

  //Solving of the system
  timeTOsolve2.stop();
  timesolve2.start();
  free_AfterSmooth_Pos_Sparse = solverCholesky.solve(RightSide);
  free_AfterSmooth_Pos = MatrixXd(free_AfterSmooth_Pos_Sparse);
  timesolve2.stop();
  timeTOEnd.start();

  //Put the positions of free_AfterSmooth_Pos in view
  igl::slice_into(free_AfterSmooth_Pos, free_vertices_indexes, 1, V);

  //Complete the base for each vertex
  computeBasis(NormalsPerVertex, T1, T2, false);

  // == Add back the details ==
  //addDetails(NormalsPerVertex, T1, T2, Difference);
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

  timeTOEnd.stop();
  timeTotal.stop();

  if (showTime)
  {
    printTime();
  }

  if (showFrameRate)
  {
    printFrameRate();
  }
}

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

bool solve2(igl::viewer::Viewer &viewer)
{
  if (status == 0)
  {
    /**** Add your code for computing the deformation from handle_vertex_positions and handle_vertices here (replace following line) ****/
    //igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);

    //REMOVE HIGH FREQUENCY DETAILS
    saveHighFrequency();

    //DEFORM SMOOTH MESH
    deformMesh();

    //ADD BACK HIGH FREQUENCY DETAILS
    addHighFrequency();
#ifdef DEBUG
    std::cout << " =============================== End of Iteration =============================== " << endl;
#endif
  }
  else
  {
    std::cerr << "Continue the process by hand instead of automatic." << endl;
  }

  return true;
};

void get_new_handle_locations()
{
  int count = 0;
  for (long vi = 0; vi < V.rows(); ++vi)
    if (handle_id[vi] >= 0)
    {
      Eigen::RowVector3f goalPosition = V.row(vi).cast<float>();
      if (handle_id[vi] == moving_handle)
      {
        if (mouse_mode == TRANSLATE)
          goalPosition += translation;
        else if (mouse_mode == ROTATE)
        {
          goalPosition -= handle_centroids.row(moving_handle).cast<float>();
          igl::rotate_by_quat(goalPosition.data(), rotation.data(), goalPosition.data());
          goalPosition += handle_centroids.row(moving_handle).cast<float>();
        }
      }
      handle_vertex_positions.row(count++) = goalPosition.cast<double>();
    }
}

bool load_mesh(string filename)
{
  igl::read_triangle_mesh(filename, V, F);
  viewer.data.clear();
  viewer.data.set_mesh(V, F);

  viewer.core.align_camera_position(V);
  handle_id.setConstant(V.rows(), 1, -1);

  // Initialize selector
  lasso = std::unique_ptr<Lasso>(new Lasso(V, F, viewer));

  selected_v.resize(0, 1);

  return true;
}

bool callback_load_mesh(Viewer &viewer, string filename)
{
  load_mesh(filename);
  return true;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cout << "Usage assignment5_bin mesh.off>" << endl;
    load_mesh("../data/woody-lo.off");
  }
  else
  {
    // Read points and normals
    load_mesh(argv[1]);
  }

  // Plot the mesh
  viewer.callback_key_down = callback_key_down;
  viewer.callback_init = [&](igl::viewer::Viewer &viewer) {

    viewer.ngui->addGroup("Deformation Controls");

    viewer.ngui->addVariable<MouseMode>("MouseMode", mouse_mode)->setItems({"SELECT", "TRANSLATE", "ROTATE", "NONE"});

    //      viewer.ngui->addButton("ClearSelection",[](){ selected_v.resize(0,1); });
    viewer.ngui->addButton("ApplySelection", []() { applySelection(); });
    viewer.ngui->addButton("ClearConstraints", []() { handle_id.setConstant(V.rows(), 1, -1); });

    viewer.ngui->addVariable("show Time", showTime);
    viewer.ngui->addVariable("show FrameRate", showFrameRate);
    viewer.ngui->addVariable("calcul LML each pass", LMLDependsonPositions);
    viewer.ngui->addButton("Print Last Time", []() { printTime(); });
    viewer.ngui->addButton("Print Last FrameRate", []() { printFrameRate(); });

    viewer.screen->performLayout();
    return false;
  };

  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;
  viewer.callback_pre_draw = callback_pre_draw;
  viewer.callback_load_mesh = callback_load_mesh;
  viewer.callback_load_mesh = callback_load_mesh;

  viewer.data.point_size = 10;
  viewer.core.set_rotation_type(igl::viewer::ViewerCore::ROTATION_TYPE_TRACKBALL);

  viewer.launch();
}

bool callback_mouse_down(igl::viewer::Viewer &viewer, int button, int modifier)
{
  if (button == (int)igl::viewer::Viewer::MouseButton::Right)
    return false;

  down_mouse_x = viewer.current_mouse_x;
  down_mouse_y = viewer.current_mouse_y;

  if (mouse_mode == SELECT)
  {
    if (lasso->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y) >= 0)
      doit = true;
    else
      lasso->strokeReset();
  }
  else if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
    int vi = lasso->pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);
    if (vi >= 0 && handle_id[vi] >= 0) //if a region was found, mark it for translation/rotation
    {
      moving_handle = handle_id[vi];
      get_new_handle_locations();
      doit = true;
    }
  }
  return doit;
}

bool callback_mouse_move(igl::viewer::Viewer &viewer, int mouse_x, int mouse_y)
{
  if (!doit)
    return false;
  if (mouse_mode == SELECT)
  {
    lasso->strokeAdd(mouse_x, mouse_y);
    return true;
  }
  if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
    if (mouse_mode == TRANSLATE)
    {
      translation = computeTranslation(viewer,
                                       mouse_x,
                                       down_mouse_x,
                                       mouse_y,
                                       down_mouse_y,
                                       handle_centroids.row(moving_handle));
    }
    else
    {
      rotation = computeRotation(viewer,
                                 mouse_x,
                                 down_mouse_x,
                                 mouse_y,
                                 down_mouse_y,
                                 handle_centroids.row(moving_handle));
    }
    get_new_handle_locations();
#ifndef UPDATE_ONLY_ON_UP
    solve(viewer);
    down_mouse_x = mouse_x;
    down_mouse_y = mouse_y;
#endif
    return true;
  }
  return false;
}

bool callback_mouse_up(igl::viewer::Viewer &viewer, int button, int modifier)
{
  if (!doit)
    return false;
  doit = false;
  if (mouse_mode == SELECT)
  {
    selected_v.resize(0, 1);
    lasso->strokeFinish(selected_v);
    return true;
  }

  if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
#ifdef UPDATE_ONLY_ON_UP
    if (moving_handle >= 0)
      solve(viewer);
#endif
    translation.setZero();
    rotation.setZero();
    rotation[3] = 1.;
    moving_handle = -1;

    compute_handle_centroids();

    return true;
  }

  return false;
};

bool callback_pre_draw(igl::viewer::Viewer &viewer)
{
  // initialize vertex colors
  vertex_colors = Eigen::MatrixXd::Constant(V.rows(), 3, .9);

  // first, color constraints
  int num = handle_id.maxCoeff();
  if (num == 0)
    num = 1;
  for (int i = 0; i < V.rows(); ++i)
    if (handle_id[i] != -1)
    {
      int r = handle_id[i] % MAXNUMREGIONS;
      vertex_colors.row(i) << regionColors[r][0], regionColors[r][1], regionColors[r][2];
    }
  // then, color selection
  for (int i = 0; i < selected_v.size(); ++i)
    vertex_colors.row(selected_v[i]) << 131. / 255, 131. / 255, 131. / 255.;

  viewer.data.set_colors(vertex_colors);
  viewer.data.V_material_specular.fill(0);
  viewer.data.V_material_specular.col(3).fill(1);
  viewer.data.dirty |= viewer.data.DIRTY_DIFFUSE | viewer.data.DIRTY_SPECULAR;

  //clear points and lines
  viewer.data.set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));
  viewer.data.set_edges(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXi::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));

  //draw the stroke of the selection
  for (unsigned int i = 0; i < lasso->strokePoints.size(); ++i)
  {
    viewer.data.add_points(lasso->strokePoints[i], Eigen::RowVector3d(0.4, 0.4, 0.4));
    if (i > 1)
      viewer.data.add_edges(lasso->strokePoints[i - 1], lasso->strokePoints[i], Eigen::RowVector3d(0.7, 0.7, 0.7));
  }

  // update the vertex position all the time
  viewer.data.V.resize(V.rows(), 3);
  viewer.data.V << V;

  viewer.data.dirty |= viewer.data.DIRTY_POSITION;

#ifdef UPDATE_ONLY_ON_UP
  //draw only the moving parts with a white line
  if (moving_handle >= 0)
  {
    Eigen::MatrixXd edges(3 * F.rows(), 6);
    int num_edges = 0;
    for (int fi = 0; fi < F.rows(); ++fi)
    {
      int firstPickedVertex = -1;
      for (int vi = 0; vi < 3; ++vi)
        if (handle_id[F(fi, vi)] == moving_handle)
        {
          firstPickedVertex = vi;
          break;
        }
      if (firstPickedVertex == -1)
        continue;

      Eigen::Matrix3d points;
      for (int vi = 0; vi < 3; ++vi)
      {
        int vertex_id = F(fi, vi);
        if (handle_id[vertex_id] == moving_handle)
        {
          int index = -1;
          // if face is already constrained, find index in the constraints
          (handle_vertices.array() - vertex_id).cwiseAbs().minCoeff(&index);
          points.row(vi) = handle_vertex_positions.row(index);
        }
        else
          points.row(vi) = V.row(vertex_id);
      }
      edges.row(num_edges++) << points.row(0), points.row(1);
      edges.row(num_edges++) << points.row(1), points.row(2);
      edges.row(num_edges++) << points.row(2), points.row(0);
    }
    edges.conservativeResize(num_edges, Eigen::NoChange);
    viewer.data.add_edges(edges.leftCols(3), edges.rightCols(3), Eigen::RowVector3d(0.9, 0.9, 0.9));
  }
#endif
  return false;
}

bool callback_key_down(igl::viewer::Viewer &viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  if (key == 'S')
  {
    mouse_mode = SELECT;
    handled = true;
  }

  if ((key == 'T') && (modifiers == IGL_MOD_ALT))
  {
    mouse_mode = TRANSLATE;
    handled = true;
  }

  if ((key == 'R') && (modifiers == IGL_MOD_ALT))
  {
    mouse_mode = ROTATE;
    handled = true;
  }
  if (key == 'A')
  {
    applySelection();
    callback_key_down(viewer, '1', 0);
    handled = true;
  }

  if (key == '0')
  {
    Eigen::MatrixXi Input(2, 3);
    Input << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXi Output;
    Output.resize(2, 3);
    Eigen::VectorXi listHor;
    listHor.resize(2, 1);
    listHor << 1, 0;
    Eigen::VectorXi listVer;
    listVer.resize(3, 1);
    listVer << 0, 1, 2;

    std::cout << "We put : " << endl;
    std::cout << "Input : \n"
              << Input << endl;
    std::cout << "listHor : \n"
              << listHor << endl;
    std::cout << "listVer : \n"
              << listVer << endl;
    igl::slice_into(Input, listHor, listVer, Output);
    std::cout << "Output : \n"
              << Output << endl;
  }
  /*
  if (key == '1')
  {
    if (status == 0)
    {
      //REMOVE HIGH FREQUENCY DETAILS
      saveHighFrequency();
      status = 1;
    }
  }

  if (key == '2')
  {
    if (status == 1)
    {
      //DEFORM SMOOTH MESH
      deformMesh();
      status = 2;
    }
  }

  if (key == '3')
  {
    if (status == 2)
    {
      //ADD BACK HIGH FREQUENCY DETAILS
      addHighFrequency();
      status = 0;
    }
  }
  */

  viewer.ngui->refresh();
  return handled;
}

void onNewHandleID()
{
  //store handle vertices too
  int numFree = (handle_id.array() == -1).cast<int>().sum();
  int num_handle_vertices = V.rows() - numFree;
  handle_vertices.setZero(num_handle_vertices);
  handle_vertex_positions.setZero(num_handle_vertices, 3);

  //ADDED
  free_vertices_indexes.setZero(V.rows() - num_handle_vertices);
  non_handle_vertex_positions.setZero(V.rows() - num_handle_vertices, 3);

  int count = 0;
  int countNOPE = 0;
  for (long vi = 0; vi < V.rows(); ++vi)
    if (handle_id[vi] >= 0)
      handle_vertices[count++] = vi;
    else
      free_vertices_indexes[countNOPE++] = vi;

  compute_handle_centroids();

  if (!LMLDependsonPositions)
  {
    //CREATION OF Lw
    Eigen::SparseMatrix<double> L;
    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> Minvert;

    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, massMatrixType, M);
    igl::invert_diag(M, Minvert);

    LML = L * Minvert * L;

    // == Division of the Bilaplacian ==
    igl::slice(LML, free_vertices_indexes, free_vertices_indexes, Aff);
    igl::slice(LML, free_vertices_indexes, handle_vertices, Afc);
  }
}

void applySelection()
{
  int index = handle_id.maxCoeff() + 1;
  for (int i = 0; i < selected_v.rows(); ++i)
  {
    const int selected_vertex = selected_v[i];
    if (handle_id[selected_vertex] == -1)
      handle_id[selected_vertex] = index;
  }
  selected_v.resize(0, 1);

  onNewHandleID();
}

void compute_handle_centroids()
{
  //compute centroids of handles
  int num_handles = handle_id.maxCoeff() + 1;
  handle_centroids.setZero(num_handles, 3);

  Eigen::VectorXi num;
  num.setZero(num_handles, 1);
  for (long vi = 0; vi < V.rows(); ++vi)
  {
    int r = handle_id[vi];
    if (r != -1)
    {
      handle_centroids.row(r) += V.row(vi);
      num[r]++;
    }
  }

  for (long i = 0; i < num_handles; ++i)
    handle_centroids.row(i) = handle_centroids.row(i).array() / num[i];
}

//computes translation for the vertices of the moving handle based on the mouse motion
Eigen::Vector3f computeTranslation(igl::viewer::Viewer &viewer,
                                   int mouse_x,
                                   int from_x,
                                   int mouse_y,
                                   int from_y,
                                   Eigen::RowVector3d pt3D)
{
  Eigen::Matrix4f modelview = viewer.core.view * viewer.data.model;
  //project the given point (typically the handle centroid) to get a screen space depth
  Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
                                      modelview,
                                      viewer.core.proj,
                                      viewer.core.viewport);
  float depth = proj[2];

  double x, y;
  Eigen::Vector3f pos1, pos0;

  //unproject from- and to- points
  x = mouse_x;
  y = viewer.core.viewport(3) - mouse_y;
  pos1 = igl::unproject(Eigen::Vector3f(x, y, depth),
                        modelview,
                        viewer.core.proj,
                        viewer.core.viewport);

  x = from_x;
  y = viewer.core.viewport(3) - from_y;
  pos0 = igl::unproject(Eigen::Vector3f(x, y, depth),
                        modelview,
                        viewer.core.proj,
                        viewer.core.viewport);

  //translation is the vector connecting the two
  Eigen::Vector3f translation = pos1 - pos0;
  return translation;
}

//computes translation for the vertices of the moving handle based on the mouse motion
Eigen::Vector4f computeRotation(igl::viewer::Viewer &viewer,
                                int mouse_x,
                                int from_x,
                                int mouse_y,
                                int from_y,
                                Eigen::RowVector3d pt3D)
{

  Eigen::Vector4f rotation;
  rotation.setZero();
  rotation[3] = 1.;

  Eigen::Matrix4f modelview = viewer.core.view * viewer.data.model;

  //initialize a trackball around the handle that is being rotated
  //the trackball has (approximately) width w and height h
  double w = viewer.core.viewport[2] / 8;
  double h = viewer.core.viewport[3] / 8;

  //the mouse motion has to be expressed with respect to its center of mass
  //(i.e. it should approximately fall inside the region of the trackball)

  //project the given point on the handle(centroid)
  Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
                                      modelview,
                                      viewer.core.proj,
                                      viewer.core.viewport);
  proj[1] = viewer.core.viewport[3] - proj[1];

  //express the mouse points w.r.t the centroid
  from_x -= proj[0];
  mouse_x -= proj[0];
  from_y -= proj[1];
  mouse_y -= proj[1];

  //shift so that the range is from 0-w and 0-h respectively (similarly to a standard viewport)
  from_x += w / 2;
  mouse_x += w / 2;
  from_y += h / 2;
  mouse_y += h / 2;

  //get rotation from trackball
  Eigen::Vector4f drot = viewer.core.trackball_angle.coeffs();
  Eigen::Vector4f drot_conj;
  igl::quat_conjugate(drot.data(), drot_conj.data());
  igl::trackball(w, h, float(1.), rotation.data(), from_x, from_y, mouse_x, mouse_y, rotation.data());

  //account for the modelview rotation: prerotate by modelview (place model back to the original
  //unrotated frame), postrotate by inverse modelview
  Eigen::Vector4f out;
  igl::quat_mult(rotation.data(), drot.data(), out.data());
  igl::quat_mult(drot_conj.data(), out.data(), rotation.data());
  return rotation;
}
