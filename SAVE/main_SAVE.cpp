#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>

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

#define DEBUG

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

bool CreatedAdjacency= false;
std::vector<std::vector<int>> AdjacencyList; //Or double ?

bool CreatedIndice = false;
std::vector<int> indiceNeighboor;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0, 3);
int NbLineToDisplay = 10;

Eigen::MatrixXd Difference;

int status = 0;

static void computeBasis(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2);
static void computeBasisSecond(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2);
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
Eigen::VectorXi non_handle_vertices(0, 1);
//updated positions of handle vertices, #V-HV x3
Eigen::MatrixXd non_handle_vertex_positions(0, 3);

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

static void compute0toXList(Eigen::VectorXi &listZeroToFSize, int nbRow)
{
  listZeroToFSize.resize(nbRow, 1);

  for (int i = 0; i < nbRow; i++)
  {
    listZeroToFSize[i] = i;
  }
}

static void computeMInverted(SparseMatrix<double> &M)
{
  Eigen::VectorXd AreasTMP;
  Eigen::VectorXd WeightVertex;

  //Calculate Mass matrix of the Shape (whole)
  igl::doublearea(V, F, AreasTMP); // ? Is it the size of the triangles ?

  //We transform it in a diagonal matrix
  M.resize(V.rows(), V.rows());
  WeightVertex.conservativeResize(V.rows(), 1);
  WeightVertex.setZero();

  //we compute the weight of each vertex as the weight of his adjacent faces
  for (int i = 0; i < F.rows(); i++)
  {
    //We add the surface of the current face to the 3 vertex part of this face
    WeightVertex[F.row(i)[0]] += AreasTMP[i]; // ? DIVIDED BY 2 BECAUSE DOUBLE AREA ?
    WeightVertex[F.row(i)[1]] += AreasTMP[i];
    WeightVertex[F.row(i)[2]] += AreasTMP[i];
  }

  //We compute the invert of M, with
  for (int i = 0; i < M.rows(); i++)
  {
    M.insert(i, i) = 1 / (WeightVertex(i)); //Note : we calcul M^-1
  }

  //igl::diag(WeightVertex, M);

  //QUESTION : Is it good idea to construct it by hand ?
  /*
      // Convert row sums into diagonal of sparse matrix
      Eigen::SparseMatrix<double> Adiag;
      igl::diag(Asum, Adiag);
  */

  /*
  //CALCULATION OF M^-1
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solverInvert; // TO REPLACE BY CHOLESKY
  solverInvert.compute(M);
  SparseMatrix<double> I(M.rows(),M.rows());
  I.setIdentity();
  Eigen::SparseMatrix<double> M_inv = solverInvert.solve(I); 
*/
}

static void solveSystem(SparseMatrix<double> &Left, SparseMatrix<double> &Right, SparseMatrix<double> &XSolved)
{
  //Solving of the system
  Eigen::SparseMatrix<double> notConstrainedPointsPosition;

  Eigen::SimplicialCholesky<SparseMatrix<double>, Eigen::RowMajor> solverCholesky;
  solverCholesky.compute(Left); //BiLaplacian_ii) ;
  XSolved = solverCholesky.solve(Right);

#ifdef DEBUG
  std::cout << " > Dimensions of Left side " << Left.rows() << " rows by " << Left.cols() << " cols." << endl;
  std::cout << " > Dimensions of Right side " << Right.rows() << " rows by " << Right.cols() << " cols." << endl;
  std::cout << " > Dimensions of XSolved side " << XSolved.rows() << " rows by " << XSolved.cols() << " cols." << endl;
#endif
}

/*
- On solve le système une première fois avec les points contraints ayant pour coordonnées leurs coordonnées initiales (dans V)
- Tu calcul la différence entre ce smoothed mesh et l'initiale mesh (différence de 0 pour les points contraints, d'autres chose pour les points non contraints)
- On solve le système une deuxième fois, avev les points contraints ayant oour coordonnées leurs nouvelles coordonnés (dans handle_vertex_positions)
- Tu rajoutes les valeurs sauvées en 2)
*/

igl::MassMatrixType massMatrixType = igl::MASSMATRIX_TYPE_DEFAULT; // MASSMATRIX_TYPE_VORONOI ?

static void saveHighFrequency()
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
  Eigen::SparseMatrix<double> LML;
  LML = L * Minvert * L;  // WE HAVE TO INVERT M ! TO DEBUG

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

  //Get old positions
  Eigen::MatrixXd HandlesPrevPosFil;
  Eigen::MatrixXd non_HandlesPrevPosFil;

  //Resize
  HandlesPrevPosFil.resize(handle_vertices.rows(), 3);
  non_HandlesPrevPosFil.resize(non_handle_vertices.rows(), 3);

  //Get only the interseting lines of V
  igl::slice(V, handle_vertices, 1, HandlesPrevPosFil);
  igl::slice(V, non_handle_vertices, 1, non_HandlesPrevPosFil);

  //Get Sparse version (dumb but necessary for solving)
  Eigen::SparseMatrix<double> HandlesPrevPos;     //The previous positions of Handles (before moving)
  Eigen::SparseMatrix<double> non_HandlesPrevPos; //The previous positions of not-Handles (before moving/smoothing)

  HandlesPrevPos = HandlesPrevPosFil.sparseView();
  non_HandlesPrevPos = non_HandlesPrevPosFil.sparseView();

#ifdef DEBUG
  std::cout << "HandlesPrevPos Overview : " << endl;
  Eigen::MatrixXd printableHandlesPrevPos = MatrixXd(HandlesPrevPos);
  for (int i = 0; i < printableHandlesPrevPos.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printableHandlesPrevPos.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printableHandlesPrevPos.row(0).size(); j++)
      {
        std::cout << printableHandlesPrevPos.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printableHandlesPrevPos Overview" << endl;

  std::cout << "non_HandlesPrevPosFil Overview : " << endl;
  Eigen::MatrixXd printablenon_HandlesPrevPosFil = MatrixXd(non_HandlesPrevPosFil);
  for (int i = 0; i < printablenon_HandlesPrevPosFil.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > printablenon_HandlesPrevPosFil.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < printablenon_HandlesPrevPosFil.row(0).size(); j++)
      {
        std::cout << printablenon_HandlesPrevPosFil.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of printablenon_HandlesPrevPosFil Overview" << endl;

  std::cout << " > Dimensions of HandlesPrevPos (lines of V corresponding to handles) " << HandlesPrevPos.rows() << " rows by " << HandlesPrevPos.cols() << " cols." << endl;
  std::cout << " > Dimensions of non_HandlesPrevPos (lines of V corresponding to NOT handles) " << non_HandlesPrevPos.rows() << " rows by " << non_HandlesPrevPos.cols() << " cols." << endl;
#endif

  //Division of the Bilaplacian
  Eigen::SparseMatrix<double> Aff;
  Eigen::SparseMatrix<double> Afc;
  Eigen::SparseMatrix<double> Acf;
  Eigen::SparseMatrix<double> Acc;

  igl::slice(LML, non_handle_vertices, non_handle_vertices, Aff);
  igl::slice(LML, non_handle_vertices, handle_vertices, Afc);
  //igl::slice(LML, handle_vertices, non_handle_vertices, Acf);   //Not useful
  //igl::slice(LML, handle_vertices, handle_vertices, Acc);       //Not useful

#ifdef DEBUG
  std::cout << " > Dimensions of Aff " << Aff.rows() << " rows by " << Aff.cols() << " cols." << endl;
  std::cout << " > Dimensions of Afc " << Afc.rows() << " rows by " << Afc.cols() << " cols." << endl;
  std::cout << " > Dimensions of Acf " << Acf.rows() << " rows by " << Acf.cols() << " cols." << endl;
  std::cout << " > Dimensions of Acc " << Acc.rows() << " rows by " << Acc.cols() << " cols." << endl;

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
  RightSide = -1 * Afc * HandlesPrevPos;

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
  Eigen::SparseMatrix<double> XSolved;
  solveSystem(Aff, RightSide, XSolved);

  Eigen::MatrixXd freePoints = MatrixXd(XSolved);
  assert(freePoints.cols() == 3);

#ifdef DEBUG
  std::cout << " freePoints Overview : " << endl;
  for (int i = 0; i < freePoints.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > freePoints.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < freePoints.row(0).size(); j++)
      {
        std::cout << freePoints.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of freePoints Overview" << endl;
#endif

  //Put the positions of freepoints in view
  igl::slice_into(freePoints, non_handle_vertices, 1, V); // SURE TO DO IT BEFORE CALCULATION OF BASIS ? YES : we need to get the old psoitions thanks to the new

  //==> GET HIGH FREQUENCY DETAILS
  //Compute the normal of each vertex
  Eigen::MatrixXd NormalsPerVertex;
  igl::per_vertex_normals(V, F, NormalsPerVertex);

  //Complete the base for each vertex
  Eigen::MatrixXd T1;
  Eigen::MatrixXd T2;
  computeBasis(NormalsPerVertex, T1, T2);

  //Get the details
  Eigen::MatrixXd freePrevPos = MatrixXd(non_HandlesPrevPos);
  computeDetails( NormalsPerVertex, T1, T2, freePrevPos, freePoints, Difference);

  //Put back the positions of the freepoints
  igl::slice_into(freePrevPos, non_handle_vertices, 1, V); 

}

static void computeBasis(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2)
{
  //NOTE : No need to compute it each time. Boolean only for the first time
  if (!CreatedAdjacency)
  {
    igl::adjacency_list(F, AdjacencyList);
    CreatedAdjacency= true;
  }

  T1.resize(V.rows(), 3);
  T2.resize(V.rows(), 3);

  for (int i = 0; i < V.rows(); i++)
  {
    //Get the first vector
    Eigen::Matrix<double, 1, 3> x = NormalsPerVertex.row(i).normalized().transpose();

    /*
    //Take the "best" neighboor
    int max = 0;
    for(int i=0 ; i <AdjacencyList.size(); i ++){
      if( ... ){
        max = ...
      }
    }*/

    //For now, just always take the first one :
    int neighboorVertexID = AdjacencyList[i][1];
    Eigen::Matrix<double, 1, 3> tmpNeighboorPos = (V.row(neighboorVertexID)-V.row(i)).transpose();
    Eigen::Matrix<double, 1, 3> projectionYonX = tmpNeighboorPos.dot(x)*x;
    Eigen::Matrix<double, 1, 3> y = (tmpNeighboorPos - projectionYonX).normalized();

    if(!CreatedIndice){
      //Store the ID we use for the neighboor
      indiceNeighboor.push_back(neighboorVertexID);
    }

    //Get the third vector
    Eigen::Matrix<double, 1, 3> z = x.cross(y).normalized();
    
    //Store the vectors
    NormalsPerVertex.row(i) = x.transpose();
    T1.row(i) = y.transpose();
    T2.row(i) = z.transpose();

#ifdef DEBUG
  std::cout << " Vector chosen for " << i << " vertice, x= " << x << " y= " << y << " z= " << z << endl;
#endif
  }

  CreatedIndice = true;
}

static void computeBasisSecond(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2){
    //NOTE : No need to compute it each time. Boolean only for the first time
  assert(CreatedAdjacency);
  assert(CreatedIndice);
  
  T1.resize(V.rows(), 3);
  T2.resize(V.rows(), 3);

  for (int i = 0; i < V.rows(); i++)
  {
    //Get the first vector
    Eigen::Matrix<double, 1, 3> x = NormalsPerVertex.row(i).normalized().transpose();

    //Take the stored neighboor
    int neighboorVertexID = indiceNeighboor[i];

    //Calculate the second vector
    Eigen::Matrix<double, 1, 3> tmpNeighboorPos = V.row(neighboorVertexID).transpose()-V.row(i).transpose();
    Eigen::Matrix<double, 1, 3> projectionYonX = tmpNeighboorPos.dot(x)*x;
    Eigen::Matrix<double, 1, 3> y = (tmpNeighboorPos - projectionYonX).normalized();

    //Get the third vector
    Eigen::Matrix<double, 1, 3> z = x.cross(y).normalized();
    
    //Store the vectors
    NormalsPerVertex.row(i) = x.transpose();
    T1.row(i) = y.transpose();
    T2.row(i) = z.transpose();

#ifdef DEBUG
  std::cout << " Vector chosen for " << i << " vertice, x= " << x << " y= " << y << " z= " << z << endl;
#endif
  }

  //Reset for next loop
  CreatedIndice = false;

}


static void computeDetails(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, Eigen::MatrixXd &BeforePos, Eigen::MatrixXd &AfterPos, Eigen::MatrixXd &Difference)
{
  assert(NormalsPerVertex.rows()==V.rows());
  assert(T1.rows()== V.rows());
  assert(T2.rows()== V.rows());
  assert(BeforePos.rows()== non_handle_vertices.rows());
  assert(AfterPos.rows()== non_handle_vertices.rows());

  //Calcul in global basis
  Difference.resize(non_handle_vertices.rows(), 3);
  Difference = BeforePos - AfterPos; //invert ? 

  Eigen::VectorXd TMP_Diff(3,1);
  Eigen::VectorXd TMP_X(3,1);
  Eigen::VectorXd TMP_Y(3,1);
  Eigen::VectorXd TMP_Z(3,1);
  int indiceFree = 0;

  for(int i=0 ; i < non_handle_vertices.rows() ; i++){
    indiceFree = non_handle_vertices[i];

    TMP_Diff = Difference.row(i).transpose();
    
    TMP_X = T1.row(indiceFree).transpose();
    TMP_Y = T2.row(indiceFree).transpose();
    TMP_Z = NormalsPerVertex.row(indiceFree).transpose();

    // Calculate the project (T1 = X, T2 = Y, Normals = Z)
    //Store it
    Difference.row(i)[0] = TMP_Diff.dot(TMP_X);
    Difference.row(i)[1] = TMP_Diff.dot(TMP_Y);
    Difference.row(i)[2] = TMP_Diff.dot(TMP_Z);
  }
}

static void addDetails(Eigen::MatrixXd &NormalsPerVertex, Eigen::MatrixXd &T1, Eigen::MatrixXd &T2, Eigen::MatrixXd &Difference){
  assert(NormalsPerVertex.rows()==V.rows());
  assert(T1.rows()== V.rows());
  assert(T2.rows()== V.rows());

  assert(Difference.rows()== non_handle_vertices.rows());

  //Calcul in global basis
  Eigen::VectorXd TMP_Diff(3,1);
  Eigen::VectorXd TMP_X(3,1);
  Eigen::VectorXd TMP_Y(3,1);
  Eigen::VectorXd TMP_Z(3,1);
  int indiceFree = 0;

  for(int i=0 ; i < non_handle_vertices.rows() ; i++){
    indiceFree = non_handle_vertices[i];

    TMP_Diff = Difference.row(i).transpose();
    
    TMP_X = T1.row(indiceFree).transpose();
    TMP_Y = T2.row(indiceFree).transpose();
    TMP_Z = NormalsPerVertex.row(indiceFree).transpose();

    // Calculate the project (T1 = X, T2 = Y, Normals = Z)
    //Store it
    Eigen::VectorXd currentPos = V.row(indiceFree).transpose();
    V.row(indiceFree) = (currentPos + TMP_X.dot(TMP_Diff)*TMP_X + TMP_Y.dot(TMP_Diff)*TMP_Y + TMP_Z.dot(TMP_Diff)*TMP_Z).transpose() ;
  }
}


static void deformMesh()
{
  //Move the handle
  igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);

  //Solve the new system

  //CREATION OF Lw
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V, F, L); //Calculate cotan laplacian of the Shape (whole)

  // CREATION OF M
  Eigen::SparseMatrix<double> M;
  Eigen::SparseMatrix<double> Minvert;
  //computeMInverted(M);
  igl::massmatrix(V, F, massMatrixType, M);
  igl::invert_diag(M, Minvert);

  //CREATION OF the "big matrix"
  Eigen::SparseMatrix<double> LML;
  LML = L * Minvert * L; // WE HAVE TO INVERT M ! TO DEBUG

  //Division of the Bilaplacian
  Eigen::SparseMatrix<double> Aff;
  Eigen::SparseMatrix<double> Afc;

  igl::slice(LML, non_handle_vertices, non_handle_vertices, Aff);
  igl::slice(LML, non_handle_vertices, handle_vertices, Afc);

  //Get only the interseting lines of V
  //Eigen::MatrixXd HandlesNextPosFil;
  //HandlesNextPosFil.resize(handle_vertices.rows(), 3);
  //igl::slice(V, handle_vertices, 1, HandlesNextPosFil);

  //Get Sparse version (dumb but necessary for solving)
  //Eigen::SparseMatrix<double> HandlesNextPos;     //The previous positions of Handles (before moving)
  //HandlesNextPos = HandlesNextPosFil.sparseView();
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
  Eigen::SparseMatrix<double> XSolved;
  solveSystem(Aff, RightSide, XSolved);

  Eigen::MatrixXd freePoints = MatrixXd(XSolved);
  assert(freePoints.cols() == 3);

#ifdef DEBUG
  std::cout << " freePoints Overview : " << endl;
  for (int i = 0; i < freePoints.rows(); i++) //NbLineToDisplay first and last lines
  {
    if (i < NbLineToDisplay || i > freePoints.rows() - NbLineToDisplay)
    {
      for (int j = 0; j < freePoints.row(0).size(); j++)
      {
        std::cout << freePoints.row(i)[j] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "> End of freePoints Overview" << endl;
#endif

  //Put the positions of freepoints in view
  igl::slice_into(freePoints, non_handle_vertices, 1, V);
}

static void addHighFrequency()
{
  //==> GET HIGH FREQUENCY DETAILS

  //Compute the normal of each vertex
  Eigen::MatrixXd NormalsPerVertex;
  igl::per_vertex_normals(V, F, NormalsPerVertex);

  //Complete the base for each vertex
  Eigen::MatrixXd T1;
  Eigen::MatrixXd T2;
  computeBasisSecond(NormalsPerVertex, T1, T2);

  //Get the details
  addDetails(NormalsPerVertex, T1, T2, Difference);
}

bool solve(igl::viewer::Viewer &viewer)
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
    std::cout << " =============================== End of Iteration =============================== " << endl;
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

    viewer.screen->performLayout();
    return false;
  };

  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;
  viewer.callback_pre_draw = callback_pre_draw;
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
  non_handle_vertices.setZero(V.rows() - num_handle_vertices);
  non_handle_vertex_positions.setZero(V.rows() - num_handle_vertices, 3);

  int count = 0;
  int countNOPE = 0;
  for (long vi = 0; vi < V.rows(); ++vi)
    if (handle_id[vi] >= 0)
      handle_vertices[count++] = vi;
    else
      non_handle_vertices[countNOPE++] = vi;

  compute_handle_centroids();
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
