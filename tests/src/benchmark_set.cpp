#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>

#include <list>
#include <vector>
#include <iostream>
#include <set>
#include <unordered_set>
#include <algorithm>

#include "ticker.h"

int main(){
    vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
    reader->SetFileName("../data/box200x200.vtu");
    reader->Update();

    vtkUnstructuredGrid *ug = reader->GetOutput();
    int NCells = ug->GetNumberOfCells();
    int NPoints = ug->GetNumberOfPoints();
    std::vector<int> ENList;
    for(int i=0;i<NCells;i++){
        for(int j=0;j<3;j++){
            ENList.push_back(ug->GetCell(i)->GetPointId(j));
        }
    }
    reader->Delete();

    // Test 1 - std::set
    std::vector< std::set<int> > NNList1(NPoints);
    double tic = get_wtime();
    for(int i=0;i<NCells;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                NNList1[ENList[i*3+j]].insert(ENList[i*3+k]);
            }
        }
    }
    std::cout<<"time std::set "<<get_wtime()-tic<<std::endl;
    for(std::set<int>::const_iterator it=NNList1[0].begin();it!=NNList1[0].end();++it)
        std::cout<<*it<<" ";
    std::cout<<endl;

    // Test 2 - std::unordered_set
    std::vector< std::unordered_set<int> > NNList2(NPoints);
    tic = get_wtime();
    for(int i=0;i<NCells;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                NNList2[ENList[i*3+j]].insert(ENList[i*3+k]);
            }
        }
    }
    std::cout<<"time std::unordered_set "<<get_wtime()-tic<<std::endl;
    for(std::unordered_set<int>::const_iterator it=NNList2[0].begin();it!=NNList2[0].end();++it)
        std::cout<<*it<<" ";
    std::cout<<std::endl;

    // Test 3 - std::vector
    std::vector< std::vector<int> > NNList3(NPoints);
    tic = get_wtime();
    for(int i=0;i<NCells;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                NNList3[ENList[i*3+j]].push_back(ENList[i*3+k]);
            }
        }
    }
    for(int i=0;i<NPoints;i++){
        std::sort(NNList3[i].begin(),NNList3[i].end());
        NNList3[i].erase(std::unique(NNList3[i].begin(), NNList3[i].end()), NNList3[i].end());
    }
    std::cout<<"time std::vector "<<get_wtime()-tic<<std::endl;
    for(std::vector<int>::const_iterator it=NNList3[0].begin();it!=NNList3[0].end();++it)
        std::cout<<*it<<" ";
    std::cout<<std::endl;

    // Test 4 - std::list
    std::vector< std::list<int> > NNList4(NPoints);
    tic = get_wtime();
    for(int i=0;i<NCells;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                NNList4[ENList[i*3+j]].push_back(ENList[i*3+k]);
            }
        }
    }
    for(int i=0;i<NPoints;i++){
        NNList4[i].unique();
    }
    std::cout<<"time std::list "<<get_wtime()-tic<<std::endl;
    for(std::list<int>::const_iterator it=NNList4[0].begin();it!=NNList4[0].end();++it)
        std::cout<<*it<<" ";
    std::cout<<std::endl;

    return 0;
}

