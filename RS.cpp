#ifndef SMTH
#define SMTH

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <sstream>
#include <DocStream.hpp>
#include <BasicDocStream.hpp>
#include "IndexManager.hpp"
#include "ResultFile.hpp"
//#include "DocUnigramCounter.hpp"
#include "RetMethod.h"
#include "QueryDocument.hpp"
#include <sstream>

#include <time.h>

#include "Parameters.h"

#include <iomanip>//for setprecisoin

#include <PorterStemmer.hpp>
#include <KStemmer.hpp>

using namespace lemur::api;
using namespace lemur::langmod;
using namespace lemur::parse;
using namespace lemur::retrieval;
using namespace std;


#define RETMODE RSMethodHM//LM(0) ,RS(1), NegKLQTE(2),NegKL(3)
#define NEGMODE negGenModeHM//coll(0) ,NonRel(1)
#define FBMODE feedbackMode//NoFB(0),NonRel(1),Normal(2),Mixture(3)
#define UPDTHRMODE 1//updatingThresholdMode//No(0),Linear(1) ,Diff(2)

template <typename T>
string numToStr(T number)
{
    ostringstream s;
    s << number;
    return s.str();
}

map <string,vector<pair<string, double> > >dictionary;
void loadDictionary();

void loadJudgment();
void computeRSMethods(Index *, Index *indFr);
void MonoKLModel(Index* ind);
vector<int> queryDocList(Index* ind,TextQueryRep *textQR);
void readWordEmbeddingFile(Index *);
void writeDocs2File(Index*);
void showNearerTermInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind);
bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
void showNearerTerms2QueryVecInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind, int avgOrMax);
void computeQueryAvgVec(Document *d,RetMethod *myMethod );
void computeMixtureForDocsAndWriteToFile(Index *ind,RetMethod *myMethod);
void readDocIdKeyWords();
void initJudgDocsVector(Index* ind,vector<int>&rel , vector<int>&nonRel,string queryID);
void readStopWord(Index *ind);

extern double startThresholdHM , endThresholdHM , intervalThresholdHM ;
extern int WHO;// 0--> server , 1-->Mozhdeh, 2-->AP, other-->Hossein
extern string outputFileNameHM;
extern string resultFileNameHM;
extern int feedbackMode;
extern double startNegWeight,endNegWeight , negWeightInterval;
extern double startNegMu, endNegMu, NegMuInterval;
extern double startDelta, endDelta, deltaInterval;
extern int RSMethodHM;
extern int negGenModeHM;
extern double smoothJMInterval1,smoothJMInterval2;
extern int updatingThresholdMode;

//int lastNewRelSize4ProfUpdating = 0, lastNewRelSize4ProfUpdatingFr = 0;


bool isRellNearest;


vector<pair<int, double> >weightedQueryTerms;

//map<string , vector<string> >queryRelDocsMap;
map<string , set<string> >queryRelDocsMap,queryRelDocsMapFr ;
map<string ,set<string> > queryNonRelDocsMap,queryNonRelDocsMapFr;

string judgmentPath,indexPath,queryPath;
string judgmentPathFr, indexPathFr, queryPathFr;
//string resultPath = "";
map<int,vector<double> >wordEmbedding;
map<int ,vector<double> >docIdKeyWords;
set<int> stopWords;

vector<pair<int ,vector<double> > > queryTermsIdVec;



#define DATASET 0 //0-->infile, 1-->ohsu
int main(int argc, char * argv[])
{


    readParams(string(argv[1]));
    cout<< "reading param file: "<<argv[1]<<endl;
    switch (WHO)
    {
    case 0:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/qrels_en";

            //indexPath= "/home/iis/Desktop/Edu/thesis/index/infile/en_notStemmed_withoutSW/index.key";
            //queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_notStemmed_en.xml";
            indexPath ="/home/iis/Desktop/Edu/thesis/index/infile/en_Stemmed_withoutSW/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_en.stemmed.xml";


        }
        break;

    case 6://server khafane:D
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/ubuntu/hrz/Data/INFILE/qrels_en";
            indexPath ="/home/ubuntu/hrz/index/infile/en_Stemmed_withoutSW/index.key";
            queryPath = "/home/ubuntu/hrz/Data/INFILE/q_en_titleKeyword_en.stemmed.xml";

        }
        break;

    default://mine
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qrels_en";
            judgmentPathFr = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qrels_fr";
            //indexPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/en_notStemmed_withoutSW/index.key";
            //queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_notStemmed_en.xml";

            indexPath ="/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/new_en_Stemmed_withoutSW/index.key";
            indexPathFr ="/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/new_fr_Stemmed_withoutSW/index.key";

            //queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_en.stemmed.xml";
            //queryPathFr = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_fr_titleKeyword_stemmed.xml";

            queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qtest_query.xml";
            queryPathFr = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qtest_query_Fr.xml";


        }
        break;
    }

    Index *ind = IndexManager::openIndex(indexPath);// without StopWord && stemmed
    Index *indFr = IndexManager::openIndex(indexPathFr);// without StopWord && stemmed


    loadDictionary();
    //return -1;

    //writeDocs2File(indFr);//from index
    //return -1;
#if 1
    //readStopWord(ind);
    //readWordEmbeddingFile(ind);
    loadJudgment();    //110,134,147 rel nadaran va hazf shan
    computeRSMethods(ind, indFr);
#endif
}

void computeRSMethods(Index* ind, Index* indFr)
{
    //clock_t tStart = clock();
    DocStream *qs = new BasicDocStream(queryPath); // Your own path to topics
    DocStream *qsFr = new BasicDocStream(queryPathFr);// tarjome shode sh ro bezar

    ArrayAccumulator accumulator(ind->docCount());
    ArrayAccumulator accumulatorFr(indFr->docCount());

    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);
    RetMethod *myMethodFr = new RetMethod(*indFr,"",accumulatorFr);


    string outFilename = outputFileNameHM;
    if(DATASET == 0)
        outFilename = outputFileNameHM+"_infile_";
    else if (DATASET == 1)
        outFilename = outputFileNameHM+"_ohsu_";

#define COMPAVG 0

    isRellNearest = false;//compute nearest from rell//used in comb..
    string methodName = "_MIX"; //RM1(c=n=100)
    outFilename += methodName;
    outFilename += "_lambda{zoj}_topPos:{10-50(20)}";//_#perQuery:{10-25(15)}";//#perQuery:{10-25(15)}//_alpha[0.1-1(0.4)]//#fb{50}_//#perQuery:{10-25(15)}////_//#topPerQueryWord:{(50,100)}////c(50,100)_//// #topPosW:30-30(0)

    ofstream out(outFilename.c_str());


    cout<< "RSMethod: "<<RSMethodHM<<" NegGenMode: "<<negGenModeHM<<" feedbackMode: "<<feedbackMode<<" updatingThrMode: "<<updatingThresholdMode<<"\n";
    cout<< "RSMethod: "<<RETMODE<<" NegGenMode: "<<NEGMODE<<" feedbackMode: "<<FBMODE<<" updatingThrMode: "<<UPDTHRMODE<<"\n";
    cout<<"outfile: "<<outFilename<<endl;

    //double oldFmeasure = 0.0 , newFmeasure = 0.0;
    double start_thresh =startThresholdHM, end_thresh= endThresholdHM;

    for (double thresh = start_thresh ; thresh<=end_thresh ; thresh += intervalThresholdHM)
        for(double fbCoef = 0.2; fbCoef <=0.91 ; fbCoef+=0.2)//lambda //5
    {
            for( double topPos = 10; topPos <= 50 ; topPos += 20 )//3//15 khube //n(50,100) for each query term//c in RM1
            {
                //for(double SelectedWord4Q = 10; SelectedWord4Q <= 25 ; SelectedWord4Q += 15)//2 //v(10,25) for each query(whole)
                {
                    //tedad feedback tuye har 2 yeksane
                    //double fbCoef =0.9;//lambda
                    //double topPos = 20;//n//c in rm1
                    double SelectedWord4Q = -1;

                    for(double c1 = 0.1 ; c1< 0.21 ;c1 += 0.1)//inc//3
                    for(double c1Fr = 0.1 ; c1Fr< 0.21 ;c1Fr += 0.1)//inc//3
                        //double c1 = 0.2, c1Fr = 0.2;
                    {
                        myMethod->setC1(c1);
                        myMethodFr->setC1(c1Fr);
                        //for(double c2 = 0.01 ; c2 < 0.08 ; c2+=0.03)//dec //3
                        double c2 = 0.04, c2Fr = 0.04;
                        {
                            //myMethod->setThreshold(init_thr);
                            myMethod->setC2(c2);
                            myMethodFr->setC2(c2Fr);
                            //for(int numOfShownNonRel = 2; numOfShownNonRel< 6; numOfShownNonRel+=3 )//2
                            int numOfShownNonRel = 1, numOfShownNonRelFr = 1;
                            {
                                for(int numOfnotShownDoc = 200 ;numOfnotShownDoc <= 401 ; numOfnotShownDoc+= 200)//3
                                for(int numOfnotShownDocFr = 200 ;numOfnotShownDocFr <= 401 ; numOfnotShownDocFr+= 200)//3
                                //int numOfnotShownDoc = 250, numOfnotShownDocFr = 250;
                                {
                                    myMethod->setThreshold(thresh);
                                    myMethodFr->setThreshold(thresh);

                                    myMethod->setTop4EachQuery(SelectedWord4Q);//v//feedbackTermCount sets
                                    myMethod->setTopWords4EachQueryTerm(topPos);//n
                                    myMethod->topsCinRM1 = topPos;//c
                                    //fr
                                    myMethodFr->setTop4EachQuery(SelectedWord4Q);//v//feedbackTermCount sets
                                    myMethodFr->setTopWords4EachQueryTerm(topPos);//n
                                    myMethodFr->topsCinRM1 = topPos;//c


                                    //myMethod->setNumberOfPositiveSelectedTopWordAndFBcount(topPos);//n
                                    //myMethod->setNumberOfTopSelectedWord4EacQword(SelectedWord4Q);//v

                                    cout<<"c1: "<<c1<<" c2: "<<c2<<" numOfShownNonRel: "<<numOfShownNonRel<<" numOfnotShownDoc: "<<numOfnotShownDoc<<" "<<endl;
                                    cout<<"FR: c1: "<<c1Fr<<" c2: "<<c2Fr<<" numOfShownNonRel: "<<numOfShownNonRelFr<<" numOfnotShownDoc: "<<numOfnotShownDocFr<<" "<<endl;

                                    //resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_c1:"+numToStr(c1)+"_c2:"+numToStr(c2)+"_#showNonRel:"+numToStr(numOfShownNonRel)+"_#notShownDoc:"+numToStr(numOfnotShownDoc)+"#topPosQT:"+numToStr(myMethod->tops4EachQueryTerm);
                                    //resultPath += "fbCoef:"+numToStr(fbCoef)+methodName+"_NoCsTuning_NoNumberT"+"_topSelectedWord:"+numToStr(SelectedWord4Q)+".res";


                                    //myMethod->setThreshold(thresh);
                                    out<<"threshold: "<<thresh<<" fbcoef: "<<fbCoef<<" n: "<<topPos<<" v: "<<SelectedWord4Q<<endl ;

                                    IndexedRealVector results;

                                    qs->startDocIteration();
                                    TextQuery *q;

                                    qsFr->startDocIteration();
                                    TextQuery *qFr;


                                    //ofstream result(resultPath.c_str());
                                    //ResultFile resultFile(1);
                                    //resultFile.openForWrite(result,*ind);

                                    //double relRetCounter = 0 , retCounter = 0 , relCounter = 0;
                                    vector<double> queriesPrecision,queriesRecall;
                                    vector<double> queriesPrecisionFr,queriesRecallFr;

                                    myMethod->setCoeffParam(fbCoef);
                                    //fr
                                    myMethodFr->setCoeffParam(fbCoef);


                                    while(qs->hasMore() && qsFr->hasMore())
                                    {
                                        myMethod->collNearestTerm.clear();
                                        myMethodFr->collNearestTerm.clear();

                                        myMethod->setThreshold(thresh);
                                        myMethodFr->setThreshold(thresh);

                                        //lastNewRelSize4ProfUpdating = 0;
                                        //lastNewRelSize4ProfUpdatingFr = 0;

                                        int numberOfNotShownDocs = 0, numberOfShownNonRelDocs = 0;
                                        int numberOfNotShownDocsFr = 0, numberOfShownNonRelDocsFr = 0;

                                        vector<int> relJudgDocs, nonRelJudgDocs;
                                        vector<int> relJudgDocsFr, nonRelJudgDocsFr;


                                        results.clear();

                                        Document *d = qs->nextDoc();
                                        q = new TextQuery(*d);
                                        QueryRep *qr = myMethod->computeQueryRep(*q);
                                        //fr
                                        Document *dFr = qsFr->nextDoc();
                                        qFr = new TextQuery(*dFr);
                                        QueryRep *qrFr = myMethodFr->computeQueryRep(*qFr);


                                        cout<<"qid: "<<q->id()<<"qidFr: "<<qFr->id()<<endl;


                                        ///*******************************************************///
#if COMPAVG
                                        computeQueryAvgVec(d,myMethod);
#endif
                                        ///*******************************************************///


                                        //vector<string> relDocs;
                                        set<string> relDocs, relDocsFr;
                                        double relDocsSize,relDocsSizeFr;

                                        map<string , set<string> >::iterator fit = queryRelDocsMap.find(q->id());
                                        map<string , set<string> >::iterator fitfr = queryRelDocsMapFr.find(qFr->id());
                                        bool hasRelJudge =  false;
                                        if( fit != queryRelDocsMap.end() )//find it!
                                        {
                                            relDocs = fit->second;
                                            hasRelJudge = true;
                                        }
                                        if( fitfr != queryRelDocsMapFr.end() )
                                        {
                                            relDocsFr = fitfr->second;
                                            hasRelJudge = true;
                                        }
                                        if(!hasRelJudge)
                                        {
                                            cerr<<"*******this query has no rel judg(ignore)**********\n";
                                            continue;
                                        }
                                        relDocsSize   = relDocs.size();
                                        relDocsSizeFr = relDocsFr.size();

                                        //for(int docID = 1 ; docID < ind->docCount() ; docID++){ //compute for all doc
                                        vector<int> docids = queryDocList(ind,((TextQueryRep *)(qr)));
                                        vector<int> docidsFr = queryDocList(indFr,((TextQueryRep *)(qrFr)));


                                        cerr<<"reldoc size: "<<relDocsSize<<" Fr "<<relDocsSizeFr<<endl;
                                        cerr<<"docs have qt size: "<<docids.size()<<" Fr "<<docidsFr.size()<<endl;


                                        int docSize = docids.size()+docidsFr.size();

                                        double resultsEn = 0 , resultsFr = 0;

                                        //for(int docID = 1 ; docID < ind->docCount() ; docID++)
                                        int ii = 0 , jj = 0;
                                        for(int i = 0 ; i< docSize; i++) //compute for docs which have queryTerm
                                        {
                                            //cerr<<docSize<<" "<<docids.size()<<" "<<docidsFr.size()<<" "<<i<<" "<<ii<<" "<<jj<<endl;

                                            bool isFr = false;
                                            int docID = 0;
                                            if(ind->document(docids[ii]) < indFr->document(docidsFr[jj]) )
                                            {
                                                docID = docids[ii];
                                                ii++;
                                            }
                                            else
                                            {
                                                isFr = true;
                                                docID = docidsFr[jj];
                                                jj++;
                                            }

                                            double sim = 0, methodThr=0;
                                            if (!isFr)
                                            {
                                                sim = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,docID, relJudgDocs, nonRelJudgDocs, isFr);
                                                methodThr = myMethod->getThreshold();
                                            }
                                            else
                                            {
                                                sim = myMethodFr->computeProfDocSim(((TextQueryRep *)(qrFr)) ,docID, relJudgDocsFr, nonRelJudgDocsFr, isFr);
                                                methodThr = myMethodFr->getThreshold();
                                            }


                                            if(sim >=  methodThr )
                                            {


                                                bool found = false;

                                                set<string>::iterator hfit;
                                                if(!isFr)
                                                {
                                                    numberOfNotShownDocs=0;

                                                    resultsEn+=1;

                                                    hfit = relDocs.find(ind->document(docID) );
                                                    if( hfit != relDocs.end() )//found
                                                    {
                                                        relDocs.erase(hfit);
                                                        relJudgDocs.push_back(docID);
                                                        found =true;
                                                    }
                                                }
                                                else //isFr
                                                {
                                                    numberOfNotShownDocsFr=0;

                                                    resultsFr+=1;

                                                    hfit = relDocsFr.find(indFr->document(docID) );
                                                    if( hfit != relDocsFr.end() )//found
                                                    {
                                                        relDocsFr.erase(hfit);
                                                        relJudgDocsFr.push_back(docID);

                                                        found =true;
                                                    }
                                                }

                                                if(!found)//not found!
                                                {
                                                    if(!isFr)
                                                    {
                                                        numberOfShownNonRelDocs++;
                                                        if( numberOfShownNonRelDocs == numOfShownNonRel )
                                                        {
                                                            myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,0);//inc thr
                                                            numberOfShownNonRelDocs = 0;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        numberOfShownNonRelDocsFr++;
                                                        if( numberOfShownNonRelDocsFr == numOfShownNonRelFr )
                                                        {
                                                            myMethodFr->updateThreshold(*((TextQueryRep *)(qrFr)), relJudgDocsFr , nonRelJudgDocsFr ,0);//inc thr
                                                            numberOfShownNonRelDocsFr = 0;
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    //delete needed????
                                                     if(!isFr)
                                                         myMethod->updateProfile(*((TextQueryRep *)(qr)),relJudgDocs , nonRelJudgDocs );
                                                     else
                                                         myMethodFr->updateProfile(*((TextQueryRep *)(qrFr)),relJudgDocsFr , nonRelJudgDocsFr );
                                                }

                                                results.PushValue(docID , sim);

                                                if(results.size() > 200)
                                                {
                                                    cout<<"BREAKKKKKKKKKK because of results size > 200\n";
                                                    break;
                                                }

                                            }
                                            else
                                            {
                                                if(!isFr)
                                                {
                                                    numberOfNotShownDocs++;

                                                    if(numberOfNotShownDocs == numOfnotShownDoc)//not show anything after |numOfnotShownDoc| docs! -->dec(thr)
                                                    {

                                                        myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,1);//dec thr
                                                        numberOfNotShownDocs = 0;
                                                    }
                                                }
                                                else
                                                {
                                                    numberOfNotShownDocsFr++;

                                                    if(numberOfNotShownDocsFr == numOfnotShownDocFr)//not show anything after |numOfnotShownDoc| docs! -->dec(thr)
                                                    {

                                                        myMethodFr->updateThreshold(*((TextQueryRep *)(qrFr)), relJudgDocsFr , nonRelJudgDocsFr ,1);//dec thr
                                                        numberOfNotShownDocsFr = 0;
                                                    }
                                                }

                                            }


                                        }//endfor docs

                                        cerr<<"\nresults size : "<<results.size()<<endl;



                                        //results.Sort();
                                        //resultFile.writeResults(q->id() ,&results,results.size());
                                        //relRetCounter += relJudgDocs.size() ;
                                        //retCounter += results.size();
                                        //relCounter += relDocsSize ;



                                        if(results.size() != 0)
                                        {
                                            queriesPrecision.push_back((double)(relJudgDocs.size() ) / resultsEn);
                                            queriesRecall.push_back((double)(relJudgDocs.size() )/ (relDocsSize) );

                                            cerr<<relJudgDocsFr.size()<<" "<<resultsFr<<" "<<relDocsSizeFr<<endl;

                                            queriesPrecisionFr.push_back((double)( relJudgDocsFr.size() )/ resultsFr );
                                            queriesRecallFr.push_back((double)( relJudgDocsFr.size() )/ (relDocsSizeFr) );
                                        }else // have no suggestion for this query
                                        {
                                            queriesPrecision.push_back(0.0);
                                            queriesRecall.push_back(0.0);

                                            queriesPrecisionFr.push_back(0.0);
                                            queriesRecallFr.push_back(0.0);
                                        }



                                        //delete d;
                                        delete q;
                                        delete qr;

                                        delete qFr;
                                        delete qrFr;


                                    }//end queries


                                    double avgPrec = 0.0 , avgRecall = 0.0;
                                    for(int i = 0 ; i < queriesPrecision.size() ; i++)
                                    {
                                        avgPrec+=queriesPrecision[i];
                                        avgRecall+= queriesRecall[i];
                                        //out<<"Prec["<<i<<"] = "<<queriesPrecision[i]<<"\tRecall["<<i<<"] = "<<queriesRecall[i]<<endl;

                                    }
                                    avgPrec/=queriesPrecision.size();
                                    avgRecall/=queriesRecall.size();



                                    /**************/
                                    //out<<"FRENCH\n";
                                    double avgPrecFr = 0.0, avgRecallFr = 0.0;
                                    for(int i = 0 ; i < queriesPrecisionFr.size() ; i++)
                                    {
                                        avgPrecFr+=queriesPrecisionFr[i];
                                        avgRecallFr+= queriesRecallFr[i];
                                        //out<<"Prec["<<i<<"] = "<<queriesPrecisionFr[i]<<"\tRecall["<<i<<"] = "<<queriesRecallFr[i]<<endl;
                                    }
                                    avgPrecFr/=queriesPrecisionFr.size();
                                    avgRecallFr/=queriesRecallFr.size();


                                    for(int i = 0 ; i < queriesPrecision.size() ; i++)
                                    {
                                        out<<"Prec["<<i<<"] = "<<(queriesPrecision[i]+queriesPrecisionFr[i])/2.0<<"\tRecall["<<i<<"] = "<<(queriesRecall[i]+queriesRecallFr[i])/2.0<<"\t";
                                        out<<"Prec["<<i<<"] = "<<queriesPrecision[i]<<"\tRecall["<<i<<"] = "<<queriesRecall[i]<<"\t";
                                        out<<"Prec["<<i<<"] = "<<queriesPrecisionFr[i]<<"\tRecall["<<i<<"] = "<<queriesRecallFr[i]<<endl;
                                    }

                                    out<<"C1: "<< c1<<"\nC2: "<<c2<<endl;
                                    out<<"numOfShownNonRel: "<<numOfShownNonRel<<"\nnumOfnotShownDoc: "<<numOfnotShownDoc<<endl;

                                    out<<"C1: "<< c1Fr<<"\nC2: "<<c2Fr<<endl;
                                    out<<"numOfShownNonRel: "<<numOfShownNonRelFr<<"\nnumOfnotShownDoc: "<<numOfnotShownDocFr<<endl;

                                    double AVGP = (avgPrec+avgPrecFr)/2.0;
                                    double AVGR = (avgRecall+avgRecallFr)/2.0;
                                    out<<"Avg Precision: "<<AVGP<<"\t"<<avgPrec<<"\t"<<avgPrecFr<<endl;
                                    out<<"Avg Recall: "<<AVGR<<"\t"<<avgRecall<<"\t"<<avgRecallFr<<endl;
                                    out<<"F-measure: "<<(2*AVGP*AVGR)/(AVGP+AVGR)<<"\t"<<(2*avgPrec*avgRecall)/(avgPrec+avgRecall)<<"\t"<<(2*avgPrecFr*avgRecallFr)/(avgPrecFr+avgRecallFr) <<endl;



#if UPDTHRMODE == 1
                                }
                            }//end numOfnotShownDoc for
                        }//end numOfShownNonRel for
                    }//end c1 for
                }//end c2 for
                //}alpha
                //}beta
                //}lambda
#endif



            }//topPos
    }//coef
    //#endif

    //cerr<<(double)(clock() - tStart);


    delete qs;
    delete myMethod;
}

void initJudgDocsVector(Index *ind,vector<int>&rel , vector<int>&nonRel,string queryID)
{

    set<string> docs;
    set<string>::iterator it;
    int counter = 5;
    if( queryRelDocsMap.find(queryID) != queryRelDocsMap.end() )//find it!
    {
        docs = queryRelDocsMap[queryID];
        //rel.assign(docs.begin(),docs.begin() + numberOfInitRelDocs - 1 );
        for(it = docs.begin() ; it !=docs.end() && counter-- > 0 ;++it )
            rel.push_back(ind->document( *it));
        if( queryNonRelDocsMap.find(queryID) != queryNonRelDocsMap.end() )//find it!
        {
            docs = queryNonRelDocsMap[queryID];
            //nonRel.assign(docs.begin(),docs.begin() + numberOfInitNonRelDocs -1);
            counter = 10;
            for(it = docs.begin() ; it !=docs.end() && counter-- > 0 ;++it )
                nonRel.push_back(ind->document(*it));
        }
    }
}
void loadDictionary()
{
    ifstream infile2;
    infile2.open ("dictionary_en2fr");

    string line2;

    PorterStemmer *stemmer = new PorterStemmer();

    map<string , int> strCountMap;

    while (getline(infile2,line2))
    {
        string word;
        stringstream ss(line2);
        ss >> word;
        string stemword = stemmer->stemWord( (char*) word.c_str() );
        strCountMap[stemword] += 1;
    }
    infile2.close();

    /********************************************************/
    ifstream infile;
    infile.open ("dictionary_en2fr");

    //map <string,vector<pair<string, double> > >dictionary;

    string line;

    //bye , khodi # 0.4 , haha # 0.3 , aba 0.5
    //hi , salam goli # 0.9 , chetory # 0.1


    //PorterStemmer *stemmer = new PorterStemmer();
    //KStemmer *kstemmer = new KStemmer();

    while (getline(infile,line))
    {
        //getline(infile,line);
        string word, temp, tr;
        double prob;

        vector<pair<string, double> > trans;
        stringstream ss(line);

        ss >> word;
        ss>> temp;//comma

        while(ss >> temp)
        {
            if( temp != "#")
                tr += temp+" ";
            else
            {
                ss>> prob;

                string aaa = tr.substr(0,tr.size()-1);//remove last space
                trans.push_back(make_pair<string, double>(aaa, prob));
                //cerr<<aaa<<" "<<prob<<endl;


                tr.clear();
                ss>>temp;//","
            }


        }

        vector<pair<string, double> >upperProbs(trans.begin(), trans.end());
        string wordStem = stemmer->stemWord( (char*) word.c_str() );

        if(strCountMap[wordStem] == 1)
        {
            dictionary.insert(make_pair< string,vector<pair<string, double> > >(wordStem, upperProbs) );

            /*cerr<<"1. "<<wordStem<<": ";
            for(int i = 0 ; i < upperProbs.size();i++)
                cerr<<upperProbs[i].first<<" "<<upperProbs[i].second<<" , ";
            cerr<<endl;*/
        }
        else if(strCountMap[wordStem] > 1)
        {
            double cnt = strCountMap[wordStem];
            //cerr<<cnt<<endl;
            for(int i = 0 ; i < upperProbs.size(); i++)
                upperProbs[i].second /= cnt;

            map<string,vector<pair<string, double> > >::iterator dicit = dictionary.find(wordStem);
            if(dicit == dictionary.end())//not found
            {
                dictionary.insert(make_pair< string,vector<pair<string, double> > >(wordStem, upperProbs) );

                /*cerr<<"2. "<<wordStem<<": ";
                for(int i = 0 ; i < upperProbs.size();i++)
                    cerr<<upperProbs[i].first<<" "<<upperProbs[i].second<<" , ";
                cerr<<endl;*/
            }else//merge trans result
            {

                map<string, double>transProbMap;
                map<string, double>::iterator it;

                vector<pair<string, double> > vec = dictionary[wordStem];

                //cerr<<dictionary.size()<<" ";
                dictionary.erase(dicit);
                //cerr<<dictionary.size()<<" s: "<<vec.size();

                for(int i = 0 ; i < vec.size() ; i++)
                {
                    //cerr<<"1: "<<transProbMap[vec[i].first]<<" "<<vec[i].second<<" "<<vec[i].first<<endl;
                    transProbMap[vec[i].first] = vec[i].second ;
                    //cerr<<"2: "<<transProbMap[vec[i].first]<<" "<<vec[i].second<<" "<<vec[i].first<<endl;
                }
                for(int i = 0 ; i< upperProbs.size();i++)
                {
                    //cerr<<"3: "<<transProbMap[upperProbs[i].first]<<" "<<upperProbs[i].second<<" "<<upperProbs[i].first<<endl;
                    transProbMap[upperProbs[i].first] += upperProbs[i].second;
                    //cerr<<"4: "<<transProbMap[upperProbs[i].first]<<" "<<upperProbs[i].second<<" "<<upperProbs[i].first<<endl;
                }

                vec.clear();
                for(it = transProbMap.begin(); it != transProbMap.end() ; ++it)
                {
                    vec.push_back(make_pair<string,double>(it->first,it->second));
                }
                dictionary.insert(make_pair< string,vector<pair<string, double> > >(wordStem, vec) );


                /*cerr<<"3. "<<wordStem<<": ";
                for(int i = 0 ; i < vec.size();i++)
                    cerr<<vec[i].first<<" "<<vec[i].second<<" , ";
                cerr<<endl;*/
            }

        }
        else
            cerr<<"DARIM MAGE?!\n\n\n";

#if 0
        map <string,vector<pair<string, double> > >::iterator mapIt;
        vector<pair<string, double> > vec;

        for(mapIt = dictionary.begin() ; mapIt != dictionary.end(); ++mapIt)
        {
            vec = mapIt->second;
            trans.clear();

            /****************************************/
            if(vec.size() == 1)
            {
                string f1 = mapIt->first, f2 = vec[0].first;
                std::transform(f1.begin(), f1.end(), f1.begin(), ::tolower);
                std::transform(f2.begin(), f2.end(), f2.begin(), ::tolower);

                if(f1 != f2)
                {
                    string aaaa = stemmer->stemWord( (char*) f1.c_str() );
                    string bbbb = stemmer->stemWord( (char*) f2.c_str() );


                    if(aaaa != bbbb)
                    {
                        string wordStem = stemmer->stemWord( (char*) mapIt->first.c_str() );
                        string transStem = stemmer->stemWord( (char*) f2.c_str() );

                        vec[0].first = transStem;
                        vec[0].second = 1.0;
                        dictionary.insert(make_pair< string,vector<pair<string, double> > >(wordStem,vec) );

                        /*cerr<<"1. "<<wordStem<<" : ";
                            for(int ii = 0 ; ii < trans.size();ii++)
                            {
                                cerr<<trans[ii].first<<" "<<trans[ii].second<<" , ";
                            }
                            cerr<<endl;*/

                    }//else stem yeki mishe pas hamun bashe

                    //cerr<<" ne "<<word<<" "<<trans[0].first<<endl;
                }//else yani yekian va hamon stem khodesh ro jaygozari kon


            }
            else//trans size > 1
            {

                double max = -10,min =10;
                for(int i = 0 ; i < vec.size();i++)
                {
                    if(vec[i].second > max)
                        max = vec[i].second;
                    if(vec[i].second < min)
                        min = vec[i].second;
                }
                trans.clear();
                for(int i = 0 ; i < vec.size();i++)
                {
                    double normalizedProb = (vec[i].second - min)/(max - min);
                    trans.push_back(make_pair<string,double>( vec[i].first, normalizedProb ) );

                    cerr<<vec[i].first<<" "<<normalizedProb<<" "<<min<<" "<<max<<endl;
                }
                dictionary.insert(make_pair< string,vector<pair<string, double> > >(wordStem, trans) );

            }


        }
#endif


    }//end getline


    /*map <string,vector<pair<string, double> > >::iterator mapIt;
    cerr<<"\n\nDIC\n";
    for(mapIt = dictionary.begin() ; mapIt != dictionary.end(); ++mapIt)
    {
        cerr<<mapIt->first<<": ";
        for(int i = 0 ; i < mapIt->second.size();i++)
            cerr<<mapIt->second[i].first<<" "<<mapIt->second[i].second<<" , ";
        cerr<<endl;
    }*/



    delete stemmer;
    return;
}
void loadJudgment()
{
    int judg,temp;
    string docName,id;

    ifstream infile;
    infile.open (judgmentPath.c_str());

    string line;
    while (getline(infile,line))
    {
        stringstream ss(line);
        if(DATASET == 0)//infile
        {
            ss >> id >> temp >> docName >> judg;
            if(judg == 1)
            {

                queryRelDocsMap[id].insert(docName.substr(11));//11 harfe aval yesane tuye infile faghat
                //map<string,bool>m;m.insert("ss",false)
                //cerr<<id<<" "<<docName<<endl;
            }else
            {
                queryNonRelDocsMap[id].insert(docName);
            }


        }
    }
    infile.close();


    //110,134,147 rel nadaran--> hazf shodan
    /*map<string , vector<string> >::iterator it;
    for(it = queryRelDocsMap.begin();it!= queryRelDocsMap.end() ; ++it)
        cerr<<it->first<<endl;*/
    /*****************************************************************************/
    //fr

    infile.open (judgmentPathFr.c_str());

    while (getline(infile,line))
    {
        stringstream ss(line);
        if(DATASET == 0)//infile
        {
            ss >> id >> temp >> docName >> judg;
            if(judg == 1)
            {

                queryRelDocsMapFr[id].insert(docName.substr(11));//11 harfe aval yesane tuye infile faghat
                //map<string,bool>m;m.insert("ss",false)
                //cerr<<id<<" "<<docName<<endl;
            }else
            {
                queryNonRelDocsMapFr[id].insert(docName);
            }


        }
    }
    infile.close();

}

void computeMixtureForDocsAndWriteToFile(Index *ind ,RetMethod *myMethod)
{

    vector<int>documentIDs;
    DocStream *qs = new BasicDocStream(queryPath); // Your own path to topics
    qs->startDocIteration();
    TextQuery *q;
    while(qs->hasMore())
    {
        Document *d = qs->nextDoc();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);

        vector<int>temp = queryDocList(ind , ((TextQueryRep *)(qr)));
        documentIDs.insert(documentIDs.begin() ,temp.begin(), temp.end());

        delete q;
        delete qr;
    }
    delete qs;

    cout<<"before: "<<documentIDs.size()<<endl;
    sort( documentIDs.begin(), documentIDs.end() );
    documentIDs.erase( unique( documentIDs.begin(), documentIDs.end() ), documentIDs.end() );

    cout<<"after: "<<documentIDs.size()<<endl;


    ofstream out;
    out.open("docKeyWords_top20word.txt");
    out<<std::setprecision(14);
    for(int i = 0 ; i < documentIDs.size() ;i++)
    {
        out<<documentIDs[i]<< " ";
        vector<double> dd = myMethod->extractKeyWord(documentIDs[i]);
        for(int j = 0 ; j < dd.size() ; j++)
            out<<setprecision(14)<<dd[j]<<" ";
        out<<endl;
    }
    out.close();
}

void readDocIdKeyWords()
{
    ifstream input("docKeyWords_top20word.txt");
    if(input.is_open())
    {
        string line;
        while(getline(input ,line))
        {

            istringstream iss(line);
            int docid=0;
            iss >> docid;
            vector<double> temp;
            do
            {
                double sub;
                iss >> sub;
                temp.push_back(sub);
                //cout << "Substring: " << sub << endl;
            } while (iss);
            docIdKeyWords.insert(pair<int , vector<double> >(docid,temp));

        }

    }else
        cerr<<"docKeyWords.txt doesn't exist!!!!!!!!!";

    input.close();

}
vector<int> queryDocList(Index* ind,TextQueryRep *textQR)
{
    vector<int> docids;
    set<int> docset;
    textQR->startIteration();
    while (textQR->hasMore()) {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0){
            cerr<<"**********"<<endl;
            continue;
        }
        DocInfoList *dList = ind->docInfoList(qTerm->id());

        dList->startIteration();
        while (dList->hasMore()) {
            DocInfo *info = dList->nextEntry();
            DOCID_T id = info->docID();
            docset.insert(id);
        }
        delete dList;
        delete qTerm;
    }
    docids.assign(docset.begin(),docset.end());
    return docids;
}

void MonoKLModel(Index* ind){
    DocStream *qs = new BasicDocStream(queryPath.c_str()); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);
    IndexedRealVector results;
    qs->startDocIteration();
    TextQuery *q;

    ofstream result("res.my_ret_method");
    ResultFile resultFile(1);
    resultFile.openForWrite(result,*ind);
    PseudoFBDocs *fbDocs;
    while(qs->hasMore()){
        Document* d = qs->nextDoc();
        //d->startTermIteration(); // It is how to iterate over query terms
        //ofstream out ("QID.txt");
        //while(d->hasMore()){
        //	const Term* t = d->nextTerm();
        //	const char* q = t->spelling();
        //	int q_id = ind->term(q);
        //	out<<q_id<<endl;
        //}
        //out.close();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);
        myMethod->scoreCollection(*qr,results);
        results.Sort();
        //fbDocs= new PseudoFBDocs(results,30,false);
        //myMethod->updateQuery(*qr,*fbDocs);
        //myMethod->scoreCollection(*qr,results);
        //results.Sort();
        resultFile.writeResults(q->id(),&results,results.size());
        cerr<<"qid "<<q->id()<<endl;
        break;
    }
}
void writeDocs2File(Index *ind)
{
    map<string,string>dic;
    ifstream myfile ("all_dictionary_fr2en");

    string delimiter = "\t";
    if (myfile.is_open())
    {
        string line,key,val;
        while ( getline (myfile,line) )
        {
            key = line.substr(0, line.find(delimiter));
            val = line.substr(line.find(delimiter)+1 ,line.size()-1);
            dic.insert(make_pair<string,string>(key,val));
            cout << line <<"$"<<key<<"#a"<<val<<"#b"<<endl;
        }
        myfile.close();
    }

    ofstream outfile;
    outfile.open("infile_docs_notStemmed_withoutSW.txt");
    {
        for(int docID = 1 ; docID <= ind->docCount(); docID++)
        {
            TermInfoList *docTermInfoList =  ind->termInfoList(docID);
            docTermInfoList->startIteration();
            vector<string> doc(3*ind->docLength(docID)," ");

            while(docTermInfoList->hasMore())
            {
                TermInfo *ti = docTermInfoList->nextEntry();
                const LOC_T *poses = ti->positions();

                for(int i = 0 ; i < ti->count() ;i++)
                {

                    //doc[poses[i] ]=ind->term(ti->termID());
                    doc[poses[i]] = dic[ind->term(ti->termID())];
                }
                //delete poses;
                //delete ti;
            }

            outfile<<"<DOC>\n<DOCNO>"<<ind->document(docID)<<"</DOCNO>\n<TEXT>\n";
            for(int i = 0 ;i < doc.size();i++)
            {
                if(doc[i] != " ")
                    outfile<<doc[i]<<" ";
            }
            outfile<<endl<<endl<<"</TEXT>\n</DOC>";

            //delete docTermInfoList;
        }

    }
    outfile.close();
}
void readWordEmbeddingFile(Index *ind)
{
    //int cc=0;
    cout << "ReadWordEmbeddingFile\n";
    string line;

    ifstream in;
#if 1
    if(WHO == 0)
    {
        if(DATASET == 0)
        {
            in.open("/home/iis/Desktop/Edu/thesis/wordEmbeddingVector/infile_docs_Stemmed_withoutSW_W2V.vectors");
            //in.open("/home/iis/Desktop/RS-Framework/QE/QE/infile_docs_Stemmed_withoutSW_W2V.vectors");//server 69
        }else if(DATASET == 1)
        {
            in.open("/home/iis/Desktop/Edu/thesis/wordEmbeddingVector/ohsu_stemmed_withoutSW_vectors100.txt");
        }

    }else if(WHO == 6)
    {
        if(DATASET == 0)//infile
            in.open("/home/ubuntu/hrz/Data/infile_docs_Stemmed_withoutSW_W2V.vectors");
    }
    else
    {
        if(DATASET == 0)//infile
            in.open("/home/hossein/Desktop/IIS/Lemur/DataSets/wordEmbeddingVector/infile_docs_Stemmed_withoutSW_W2V.vectors");
        else if(DATASET == 1)//ohsu
            in.open("/home/hossein/Desktop/IIS/Lemur/DataSets/wordEmbeddingVector/ohsu_stemmed_withoutSW_vectors100.txt");
    }
    getline(in,line);//first line is statistical in W2V
#endif
#if 0
    ifstream in("/home/hossein/Desktop/IIS/Lemur/DataSets/wordEmbeddingVector/infile_vectors_100D_Glove.txt");
#endif
    while(getline(in,line))
    {
        //cc++;
        istringstream iss(line);

        string sub;
        double dd;
        iss >> sub;

        if(sub.size() <= 1)
            continue;

        int termID = ind->term(sub);

        while (iss>>dd)
            wordEmbedding[termID].push_back(dd);
    }

    cout<<"ReadWordEmbeddingFile END\n";
}


bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem)
{
    return firstElem.first > secondElem.first;
}

void showNearerTerms2QueryVecInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind, int avgOrMax)
{
    ofstream inputfile;
    inputfile.open("outputfiles/termsNearer2QueryWordsMaximum.txt");

    qs->startDocIteration();
    TextQuery *q;
    while(qs->hasMore())//queries
    {
        Document* d = qs->nextDoc();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);
        TextQueryRep *textQR = (TextQueryRep *)(qr);

        //cout<<wordEmbedding.size()<<" "<<ind->termCountUnique()<<endl;



        vector<vector<double> > queryTerms;
        double counter =0 ;
        textQR->startIteration();
        while(textQR->hasMore())
        {

            counter += 1;
            QueryTerm *qt = textQR->nextTerm();
            if(wordEmbedding.find(qt->id()) != wordEmbedding.end())
            {
                queryTerms.push_back(wordEmbedding[qt->id()]);
            }
            else
            {
                delete qt;
                continue;
            }

            inputfile<<ind->term(qt->id())<<" ";
            delete qt;
        }

        inputfile<<" : ";
        vector<double> queryAvg( myMethod->W2VecDimSize);

        if(avgOrMax == 0)
        {
            for(int i =0 ; i< queryTerms.size() ; i++)
            {
                for(int j = 0 ;j<queryTerms[i].size() ; j++)
                    queryAvg[j] += queryTerms[i][j];
            }
            for(int i = 0 ; i < queryAvg.size() ;i++)
                queryAvg[i] /= counter;
        }else if (avgOrMax == 1)
        {
            for(int i =0 ; i< queryTerms.size() ; i++)
            {
                for(int j = 0 ;j<queryTerms[i].size() ; j++)
                {
                    if(queryAvg[j] < queryTerms[i][j])
                        queryAvg[j] = queryTerms[i][j];
                }
            }

        }


        vector<double>dtemp;
        vector<pair<double,int> >simTermid;
        for(int i = 1 ; i < ind->termCountUnique() ; i++)
        {
            if(wordEmbedding.find(i) != wordEmbedding.end())
                dtemp = wordEmbedding[i];
            else
                continue;


            double sim = myMethod->cosineSim(queryAvg,dtemp);
            simTermid.push_back(pair<double,int>(sim,i));
        }
        std::sort(simTermid.begin() , simTermid.end(),pairCompare);

        for(int i = 0 ; i < 10 ; i++)
            inputfile <<"( "<< ind->term(simTermid[i].second)<<" , "<<simTermid[i].first<<" ) ";

        inputfile<<endl;
        simTermid.clear();dtemp.clear();queryAvg.clear();


        delete textQR;
        delete q;
    }

    //delete qr;
    //delete d;

    inputfile<<endl;
    inputfile.close();

}

void computeQueryAvgVec(Document *d,RetMethod *myMethod )
{

    queryTermsIdVec.clear();

    TextQuery *q = new TextQuery(*d);
    QueryRep *qr = myMethod->computeQueryRep(*q);
    TextQueryRep *textQR = (TextQueryRep *)(qr);


    const std::map<int,vector<double> >::iterator endIt = wordEmbedding.end();
    textQR->startIteration();
    while(textQR->hasMore())
    {
        QueryTerm *qt = textQR->nextTerm();
        const std::map<int,vector<double> >::iterator it = wordEmbedding.find(qt->id());

        if(it != endIt)//found
        {
            for(int i=0; i < qt->weight() ; i++)   //(<queryW1,<1,2,4>)(queryW1,<1,2,3>)
                queryTermsIdVec.push_back(make_pair<int , vector<double> > (qt->id() ,it->second ) );
        }
        else
        {
            delete qt;
            continue;
        }
        delete qt;
    }

    delete qr;
    delete q;
    //delete textQR;


    vector<double> queryAvg( myMethod->W2VecDimSize ,0.0);
    for(int i = 0 ; i< queryTermsIdVec.size() ; i++)
    {
        for(int j = 0 ; j < myMethod->W2VecDimSize; j++)
            queryAvg[j] += queryTermsIdVec[i].second[j];
    }

    double qsize = queryTermsIdVec.size();
    for(int i = 0 ; i < myMethod->W2VecDimSize/*queryAvg.size()*/ ;i++)
        queryAvg[i] /= qsize;

    myMethod->Vq.clear();
    //myMethod->Vq.assign(myMethod->W2VecDimSize ,0.0);
    //myMethod->Vq = queryAvg;
    myMethod->Vq.assign( queryAvg.begin(),queryAvg.end());

    //delete qqr;
    //delete qq;
    //delete textQR;


    //weighted Query //dist from Avg
    weightedQueryTerms.clear();
    double totalSc = 0.0;
    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        //double sc = myMethod->softMaxFunc2(queryTermsIdVec[i].second , myMethod->Vq);
        double sc = myMethod->cosineSim(queryTermsIdVec[i].second , myMethod->Vq);

        weightedQueryTerms.push_back(make_pair<int, double>(queryTermsIdVec[i].first , sc));
        totalSc +=sc;
    }
    for(int i = 0 ;i< weightedQueryTerms.size() ;i++)
        weightedQueryTerms[i].second /= totalSc;


}

void showNearerTermInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind)
{
    ofstream inputfile;
    inputfile.open("outputfiles/similar2QueryWord.txt");



    qs->startDocIteration();
    TextQuery *q;
    while(qs->hasMore())//queries
    {
        Document* d = qs->nextDoc();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);

        TextQueryRep *textQR = (TextQueryRep *)(qr);


        textQR->startIteration();
        while(textQR->hasMore())//query terms
        {
            vector<pair<double,int> >simTermid;
            vector<double> qtemp,dtemp;
            QueryTerm *qt = textQR->nextTerm();

            if(wordEmbedding.find(qt->id()) != wordEmbedding.end())
                qtemp = wordEmbedding[qt->id()];
            else
                continue;

            cout<<wordEmbedding.size()<<" "<<ind->termCountUnique()<<endl;

            for(int i =1 ; i< ind->termCountUnique() ; i++)
            {

                if(wordEmbedding.find(i) != wordEmbedding.end())

                {
                    dtemp = wordEmbedding[i];
                    //cout<<"here!\n";
                }
                else
                {
                    //cout<<"here22222!\n";
                    continue;
                }
                //if(dtemp.size() == 0 )
                //    continue;


                double sim = myMethod->cosineSim(qtemp,dtemp);
                simTermid.push_back(pair<double,int>(sim,i));
            }
            std::sort(simTermid.begin() , simTermid.end(),pairCompare);


            inputfile<<ind->term(qt->id())<<": ";
            //for(int i=simTermid.size()-1 ; i> simTermid.size()- 5;i--)
            for(int i = 0 ; i < 5 ; i++)
                inputfile <<"( "<< ind->term(simTermid[i].second)<<" , "<<simTermid[i].first<<" ) ";

            inputfile<<endl;
            delete qt;
            simTermid.clear();
            qtemp.clear();
            dtemp.clear();
        }

        delete textQR;
        delete q;
        //delete qr;
        //delete d;
    }
    inputfile<<endl;
    inputfile.close();
}

void readStopWord(Index *ind)
{
    string mterm;
    ifstream input("dataSets/stops_en.txt");
    if(input.is_open())
    {
        int cc=0;
        while(getline(input,mterm))
        {
            cc++;
            //std::cout<<mterm<<" aaa ";
            if(mterm.size()>1)
                mterm.erase(mterm.size()-1,mterm.size());
            //std::cout<<" ttt "<<mterm<<endl;
            stopWords.insert( ind->term(mterm) );
        }
        cout<<cc<<" SW size: "<<stopWords.size()<<endl;

        input.close();
    }else
    {
        cerr<<"FILE NOT OPENED";
    }
    stopWords.erase(stopWords.find(0));

}

#if 0
#include "pugixml.hpp"
using namespace spugi;
void ParseQuery(){
    ofstream out("topics.txt");
    xml_document doc;
    xml_parse_result result = doc.load_file("/home/hossein/Desktop/lemur/DataSets/Infile/Data/q_en.xml");// Your own path to original format of queries
    xml_node topics = doc.child("topics");
    for (xml_node_iterator topic = topics.begin(); topic != topics.end(); topic++){
        xml_node id = topic->child("identifier");
        xml_node title = topic->child("title");
        xml_node desc = topic->child("description");
        xml_node nar = topic->child("narrative");
        out<<"<DOC>"<<endl;
        out<<"<DOCNO>"<<id.first_child().value()<<"</DOCNO>"<<endl;
        out<<"<TEXT>"<<endl;
        out<<title.first_child().value()<<endl;
        out<<"</TEXT>"<<endl;
        out<<"</DOC>"<<endl;

    }
    printf("Query Parsed.\n");
}
#endif
#endif

