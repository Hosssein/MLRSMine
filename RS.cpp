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

map <string,vector<pair<string, double> > >dictionaryFr2En, invDictionaryFr2En;
map <string,vector<pair<string, double> > >dictionaryEn2Fr, invDictionaryEn2Fr;

void loadDictionary();

void loadJudgment();
void computeRSMethods(Index *, Index *indFr);
void MonoKLModel(Index* ind);
vector<int> queryDocList(Index* ind,TextQueryRep *textQR);
void readWordEmbeddingFile(Index *,Index *);
void writeDocs2File(Index*);
void showNearerTermInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind);
bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
void showNearerTerms2QueryVecInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind, int avgOrMax);
void computeQueryAvgVec(Document *d, RetMethod *myMethod , bool isFr);
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
double alphaTrans ,betaTrans;

vector<pair<int, double> >weightedQueryTerms, weightedQueryTermsFr;

//map<string , vector<string> >queryRelDocsMap;
map<string , set<string> >queryRelDocsMap,queryRelDocsMapFr ;
map<string ,set<string> > queryNonRelDocsMap,queryNonRelDocsMapFr;

string judgmentPath,indexPath,queryPath;
string judgmentPathFr, indexPathFr, queryPathFr;
//string resultPath = "";
map<int,vector<double> >wordEmbedding,wordEmbeddingFr;
map<int ,vector<double> >docIdKeyWords;
set<int> stopWords;

vector<pair<int ,vector<double> > > queryTermsIdVec, queryTermsIdVecFr;

void writeIndexDocs2File(Index *ind);


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
            judgmentPathFr ="/home/iis/Desktop/Edu/thesis/Data/INFILE/qrels_fr";

            //indexPath= "/home/iis/Desktop/Edu/thesis/index/infile/en_notStemmed_withoutSW/index.key";
            //queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_notStemmed_en.xml";
            indexPath ="/home/iis/Desktop/Edu/thesis/index/infile/new_en_notStemmed_withoutSW/index.key";
            indexPathFr="/home/iis/Desktop/Edu/thesis/index/infile/new_fr_notStemmed_withoutSW/index.key";

            queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_notStemmed_en.xml";
            queryPathFr="/home/iis/Desktop/Edu/thesis/Data/INFILE/q_fr_titleKeyword_notStemmed.xml";


        }
        break;

    case 6://server khafane:D
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/hosein/data/infile/qrels_en";
            judgmentPathFr = "/home/hosein/data/infile/qrels_fr";

            indexPath ="/home/hosein/index/infile/new_en_notStemmed_withoutSW/index.key";
            indexPathFr ="/home/hosein/index/infile/new_fr_notStemmed_withoutSW/index.key";

            queryPath = "/home/hosein/data/infile/q_en_titleKeyword_notStemmed_en.xml";
            queryPathFr = "/home/hosein/data/infile/q_fr_titleKeyword_notstemmed.xml";

        }
        break;

    default://mine
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qrels_en";
            judgmentPathFr = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qrels_fr";
            //indexPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/en_notStemmed_withoutSW/index.key";
            //queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_notStemmed_en.xml";

            indexPath ="/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/new_en_notStemmed_withoutSW/index.key";
            indexPathFr ="/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/new_fr_notStemmed_withoutSW/index.key";

            //queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_en.stemmed.xml";
            //queryPathFr = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_fr_titleKeyword_stemmed.xml";

            queryPath =   "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qtest_query.xml";
            queryPathFr = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qtest_query_Fr.xml";


        }
        break;
    }

    Index *ind = IndexManager::openIndex(indexPath);// without StopWord && stemmed
    Index *indFr = IndexManager::openIndex(indexPathFr);// without StopWord && stemmed

    //writeIndexDocs2File(ind);
    //return -1;
    /*ofstream out;
    out.open("fr_terms.txt");
    for(int i= 1 ; i<= indFr->termCountUnique(); i++ )
    {
        out<<indFr->term(i)<<endl;
    }
    out.close();
    return-1;*/

    loadDictionary();

    readWordEmbeddingFile(ind,indFr);
    loadJudgment();    //110,134,147 rel nadaran va hazf shan
    computeRSMethods(ind, indFr);
}

void computeRSMethods(Index* ind, Index* indFr)
{
    //2 profile e va halate trans!!

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

#define COMPAVG 1
#define RAHIMI  0
#define FANG_NEG 1
    isRellNearest = false;//compute nearest from rell//used in comb..
    string methodName = "_rahimidic_Negcoll_0.1_LLWE_alpha:0.2"; //RM1(c=n=100)
    //string methodName = "_hosseindic_Rahimi_LLWE_alpha:0.2"; //RM1(c=n=100)
    //string methodName = "noprof";//"logistic_we_tp15_exp_a:0.5_b:0.5_";
    outFilename += methodName;
    //outFilename += "_lambda{zoj}_topPos:{10-50(20)}";//_#perQuery:{10-25(15)}";//#perQuery:{10-25(15)}//_alpha[0.1-1(0.4)]//#fb{50}_//#perQuery:{10-25(15)}////_//#topPerQueryWord:{(50,100)}////c(50,100)_//// #topPosW:30-30(0)

    ofstream out(outFilename.c_str());


    cout<< "RSMethod: "<<RSMethodHM<<" NegGenMode: "<<negGenModeHM<<" feedbackMode: "<<feedbackMode<<" updatingThrMode: "<<updatingThresholdMode<<"\n";
    cout<< "RSMethod: "<<RETMODE<<" NegGenMode: "<<NEGMODE<<" feedbackMode: "<<FBMODE<<" updatingThrMode: "<<UPDTHRMODE<<"\n";
    cout<<"outfile: "<<outFilename<<endl;

    //double oldFmeasure = 0.0 , newFmeasure = 0.0;
    double start_thresh =startThresholdHM, end_thresh= endThresholdHM;

    for (double thresh = start_thresh ; thresh<=end_thresh ; thresh += intervalThresholdHM)
        //for(double fbCoef = 0.1; fbCoef <=0.91 ; fbCoef+=0.2)//lambda //5
    {
        double fbCoef =0.2;//alpha

        for(double bb = 0.0 ; bb <= 1-fbCoef; bb += 0.1)
        //for( double topPos = 15; topPos <= 15 ; topPos += 10 )//1//15 khube //n(50,100) for each query term//c in RM1
        {

            for(double SelectedWord4Q = 10; SelectedWord4Q <= 30 ; SelectedWord4Q += 10)//3 //v(10,25) for each query(whole)
            {
                //double bb =-1;
                //tedad feedback tuye har 2 yeksane
                double topPos = 15;//eachqueryTerm//n//c in rm1
                //double SelectedWord4Q = 50;//fbTermCount
                alphaTrans = fbCoef;
                betaTrans = bb;

                for(double c1 = 0.1 ; c1< 0.17 ;c1 += 0.05)//inc//2
                    //for(double c1Fr = 0.1 ; c1Fr< 0.21 ;c1Fr += 0.1)//inc//3
                    //double c1 = 0.1, c1Fr = 0.1;
                {
                    double c1Fr = c1;

                    myMethod->setC1(c1);
                    myMethodFr->setC1(c1Fr);
                    for(double c2 = 0.01 ; c2 < 0.05 ; c2+=0.03)//dec //2
                        //double c2 = 0.04, c2Fr = 0.04;
                    {
                        double c2Fr = c2;
                        //myMethod->setThreshold(init_thr);
                        myMethod->setC2(c2);
                        myMethodFr->setC2(c2Fr);
                        for(int numOfShownNonRel = 2; numOfShownNonRel< 6; numOfShownNonRel+=3 )//2
                            //int numOfShownNonRel = 1, numOfShownNonRelFr = 1;
                        {
                            int numOfShownNonRelFr = numOfShownNonRel;

                            for(int numOfnotShownDoc = 100 ;numOfnotShownDoc <= 401 ; numOfnotShownDoc+= 200)//3
                                //for(int numOfnotShownDocFr = 200 ;numOfnotShownDocFr <= 401 ; numOfnotShownDocFr+= 200)//3
                                //int numOfnotShownDoc = 250, numOfnotShownDocFr = 250;
                            {
                                int numOfnotShownDocFr = numOfnotShownDoc;


                                myMethod->setNegMu(1000);
                                myMethodFr->setNegMu(1000);
                                myMethod->setDelta(0.001);
                                myMethodFr->setDelta(0.001);
                                //myMethod->setLambda1(0.05);
                                //myMethodFr->setLambda1(0.05);
                                //myMethod->setLambda2(0.05);
                                //myMethodFr->setLambda2(0.05);


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

                                cerr<<"c1: "<<c1<<" c2: "<<c2<<" numOfShownNonRel: "<<numOfShownNonRel<<" numOfnotShownDoc: "<<numOfnotShownDoc<<" "<<endl;
                                cerr<<"FR: c1: "<<c1Fr<<" c2: "<<c2Fr<<" numOfShownNonRel: "<<numOfShownNonRelFr<<" numOfnotShownDoc: "<<numOfnotShownDocFr<<" "<<endl;

                                //resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_c1:"+numToStr(c1)+"_c2:"+numToStr(c2)+"_#showNonRel:"+numToStr(numOfShownNonRel)+"_#notShownDoc:"+numToStr(numOfnotShownDoc)+"#topPosQT:"+numToStr(myMethod->tops4EachQueryTerm);
                                //resultPath += "fbCoef:"+numToStr(fbCoef)+methodName+"_NoCsTuning_NoNumberT"+"_topSelectedWord:"+numToStr(SelectedWord4Q)+".res";


                                //myMethod->setThreshold(thresh);
                                out<<"threshold: "<<thresh<<" negCoef: "<<0.1<<" delta: "<<0.001<<" alpha: "<<fbCoef<<" beta: "<<betaTrans<<" gama: "<<1-fbCoef-betaTrans<<" pW: "<<topPos<<" pQ: "<<SelectedWord4Q<<endl ;

                                IndexedRealVector results;

                                qs->startDocIteration();
                                TextQuery *q;

                                qsFr->startDocIteration();
                                TextQuery *qFr;


                                //ofstream result(resultPath.c_str());
                                //ResultFile resultFile(1);
                                //resultFile.openForWrite(result,*ind);

                                //double relRetCounter = 0 , retCounter = 0 , relCounter = 0;
                                vector<double> queriesPrecision,queriesRecall, queriesPrecisionAll;
                                vector<double> queriesPrecisionFr,queriesRecallFr,queriesRecallAll;

                                myMethod->setCoeffParam(fbCoef);
                                //fr
                                myMethodFr->setCoeffParam(fbCoef);


                                while(qs->hasMore() && qsFr->hasMore())
                                {

                                    myMethod->collNearestTerm.clear();//combSum inas fek konam
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

                                    bool newNonRel = false,newNonRelFr = false;
#if FANG_NEG

                                    myMethod->clearPrevDistQuery();//fang
                                    myMethodFr->clearPrevDistQuery();//fang

                                    myMethod->fillQueryTermIndexes((TextQueryRep *)(qr));//FANG fang FANG
                                    myMethodFr->fillQueryTermIndexes((TextQueryRep *)(qrFr));//FANG fang FANG
#endif


                                    ///*******************************************************///
#if COMPAVG
                                    computeQueryAvgVec(d,myMethod,false);
                                    computeQueryAvgVec(dFr,myMethodFr,true);
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
                                        bool isFr = false;
                                        int docID = 0;
                                        if( ind->document(docids[ii]) < indFr->document(docidsFr[jj]) )
                                        {
                                            if(ii == docids.size())
                                            {
                                                isFr = true;
                                                docID = docidsFr[jj];
                                                jj++;
                                                jj = std::min(jj, (int)docidsFr.size());
                                            }
                                            else
                                            {
                                                docID = docids[ii];
                                                ii++;
                                                ii = std::min(ii, (int)docids.size());
                                            }

                                        }
                                        else
                                        {
                                            if(jj == docidsFr.size())
                                            {
                                                docID = docids[ii];
                                                ii++;
                                                ii = std::min(ii, (int)docids.size());
                                            }
                                            else
                                            {
                                                isFr = true;
                                                docID = docidsFr[jj];
                                                jj++;
                                                jj = std::min(jj, (int)docidsFr.size() );
                                            }
                                        }

#if 0 //begin test
                                        cerr<<"EN:\n";
                                        set<string>::iterator setit = relDocs.begin();
                                        for(setit ; setit != relDocs.end(); ++setit)
                                        {
                                            //double ss = myMethodFr->scoreDoc( *qrFr ,indFr->document(*setit) );
                                            double ss = myMethod->computeProfDocSim(((TextQueryRep *)(qr)), ind->document(*setit), relJudgDocs, nonRelJudgDocs, false, indFr, newNonRel);
                                            cerr<<ss<<" ";
                                        }

                                        cerr<<"\nFR:\n";
                                        set<string>::iterator setitFr = relDocsFr.begin();
                                        for(setitFr ; setitFr != relDocsFr.end(); ++setitFr)
                                        {
                                            //double ss = myMethodFr->scoreDoc( *qrFr ,indFr->document(*setit) );
                                            double ss = myMethodFr->computeProfDocSim(((TextQueryRep *)(qrFr)) ,indFr->document(*setitFr), relJudgDocsFr, nonRelJudgDocsFr, true, ind, newNonRelFr);
                                            cerr<<ss<<" ";
                                        }
                                        return;
#endif
#if 0//RAHIMI test
                                        //EN
                                        cerr<<"EN:\n";
                                        set<string>::iterator setit = relDocs.begin();
                                        for(setit ; setit != relDocs.end(); ++setit)
                                        {
                                            //cerr<<"en: "<<ind->document(docID)<<endl;
                                            double ss = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,ind->document(*setit), relJudgDocs, nonRelJudgDocs, false , indFr, newNonRel);
                                            cerr<<ss<<" ";
                                        }
                                        cerr<<"\nFR:\n";
                                        set<string>::iterator setitFr = relDocsFr.begin();
                                        for(setitFr ; setitFr != relDocsFr.end(); ++setitFr)
                                        {
                                            //cerr<<"en: "<<ind->document(docID)<<endl;
                                            double ss = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,indFr->document(*setitFr), relJudgDocs, nonRelJudgDocs, true, indFr, newNonRel);
                                            cerr<<ss<<" ";
                                        }
                                        return;
#endif
//end test

                                        double sim = 0, methodThr=0;
#if RAHIMI
                                        sim = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,docID, relJudgDocs, nonRelJudgDocs,isFr, indFr, newNonRel);
                                        methodThr = myMethod->getThreshold();


#else
                                        if (!isFr)
                                        {
                                            //cerr<<"en: "<<ind->document(docID)<<endl;
                                            sim = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,docID, relJudgDocs, nonRelJudgDocs, isFr, indFr, newNonRel);
                                            methodThr = myMethod->getThreshold();
                                        }
                                        else
                                        {
                                            //cerr<<"fr: "<<indFr->document(docID)<<endl;
                                            sim = myMethodFr->computeProfDocSim(((TextQueryRep *)(qrFr)) ,docID, relJudgDocsFr, nonRelJudgDocsFr, isFr, ind, newNonRelFr);
                                            methodThr = myMethodFr->getThreshold();
                                        }
#endif
                                        //sleep(1);

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
                                                    newNonRel = false;
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
                                                    newNonRelFr = false;
                                                    relDocsFr.erase(hfit);
                                                    relJudgDocsFr.push_back(docID);
                                                    found =true;
                                                }
                                            }

                                            if(!found)//not found!
                                            {
                                                if(!isFr)
                                                {
                                                    newNonRel = true;
                                                    nonRelJudgDocs.push_back(docID);

                                                    numberOfShownNonRelDocs++;
                                                    if( numberOfShownNonRelDocs == numOfShownNonRel )
                                                    {
                                                        myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,0);//inc thr
                                                        numberOfShownNonRelDocs = 0;
                                                    }
                                                }
                                                else
                                                {
                                                    newNonRelFr = true;
                                                    nonRelJudgDocsFr.push_back(docID);
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
                                                if(!isFr)//en//0
                                                {
                                                    delete qr;
                                                    qr = myMethod->computeQueryRep(*q);
                                                    //cerr<<"en1\n";
                                                    myMethod->updateProfile(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs,ind, indFr, relJudgDocsFr,isFr, myMethodFr );
                                                }
                                                else//fr//1
                                                {
                                                    //cerr<<isFr<<"fr 1\n";
                                                    delete qrFr;
                                                    qrFr = myMethodFr->computeQueryRep(*qFr);
                                                    myMethodFr->updateProfile(*((TextQueryRep *)(qrFr)),relJudgDocsFr , nonRelJudgDocsFr,indFr, ind, relJudgDocs,isFr, myMethod );
                                                }
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
                                                newNonRel = false;
                                                numberOfNotShownDocs++;

                                                if(numberOfNotShownDocs == numOfnotShownDoc)//not show anything after |numOfnotShownDoc| docs! -->dec(thr)
                                                {
                                                    myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,1);//dec thr
                                                    numberOfNotShownDocs = 0;
                                                }
                                            }
                                            else
                                            {
                                                newNonRelFr = false;
                                                numberOfNotShownDocsFr++;

                                                if(numberOfNotShownDocsFr == numOfnotShownDocFr)//not show anything after |numOfnotShownDoc| docs! -->dec(thr)
                                                {

                                                    myMethodFr->updateThreshold(*((TextQueryRep *)(qrFr)), relJudgDocsFr , nonRelJudgDocsFr ,1);//dec thr
                                                    numberOfNotShownDocsFr = 0;
                                                }
                                            }

                                        }


                                    }//endfor docs

                                    cerr<<"rel size en: "<<relJudgDocs.size()<<" fr: "<<relJudgDocsFr.size()<<endl;
                                    cerr<<"results size : "<<results.size()<<" "<<resultsEn<<" "<<resultsFr<<endl<<endl;



                                    //results.Sort();
                                    //resultFile.writeResults(q->id() ,&results,results.size());
                                    //relRetCounter += relJudgDocs.size() ;
                                    //retCounter += results.size();
                                    //relCounter += relDocsSize ;




                                    if(resultsEn != 0 && relDocsSize !=0)
                                    {
                                        queriesPrecision.push_back((double)(relJudgDocs.size() ) / resultsEn);
                                        queriesRecall.push_back((double)(relJudgDocs.size() )/ (relDocsSize) );

                                    }else
                                    {
                                        queriesPrecision.push_back(0.0);
                                        queriesRecall.push_back(0.0);
                                    }
                                    if(resultsFr != 0 && relDocsSizeFr !=0 )
                                    {
                                        queriesPrecisionFr.push_back((double)( relJudgDocsFr.size() )/ resultsFr );
                                        queriesRecallFr.push_back((double)( relJudgDocsFr.size() )/ (relDocsSizeFr) );
                                    }else
                                    {
                                        queriesPrecisionFr.push_back(0.0);
                                        queriesRecallFr.push_back(0.0);
                                    }
                                    if(results.size() != 0)
                                    {
                                        queriesPrecisionAll.push_back((double)(relJudgDocs.size()+relJudgDocsFr.size() )/((double)(resultsEn+resultsFr) ) );
                                        queriesRecallAll.push_back((double)(relJudgDocs.size()+relJudgDocsFr.size() )/((double)(relDocsSize+relDocsSizeFr) ) );

                                    }else // have no suggestion for this query
                                    {
                                        queriesPrecisionAll.push_back(0.0);
                                        queriesRecallAll.push_back(0.0);
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
                                double avgPrecAll = 0.0 , avgRecallAll =0.0;
                                double avgPrecFr = 0.0, avgRecallFr = 0.0;
                                for(int i = 0 ; i < queriesPrecisionFr.size() ; i++)
                                {
                                    avgPrecAll += queriesPrecisionAll[i];
                                    avgRecallAll += queriesRecallAll[i];

                                    avgPrecFr+=queriesPrecisionFr[i];
                                    avgRecallFr+= queriesRecallFr[i];
                                    //out<<"Prec["<<i<<"] = "<<queriesPrecisionFr[i]<<"\tRecall["<<i<<"] = "<<queriesRecallFr[i]<<endl;
                                }
                                avgPrecAll/=queriesPrecisionAll.size();
                                avgRecallAll/=queriesRecallAll.size();

                                avgPrecFr/=queriesPrecisionFr.size();
                                avgRecallFr/=queriesRecallFr.size();


                                for(int i = 0 ; i < queriesPrecision.size() ; i++)
                                {
                                    out<<"PrecAl["<<i<<"] = "<<queriesPrecisionAll[i]<<"\tRecallAl["<<i<<"] = "<<queriesRecallAll[i]<<"\t";
                                    out<<"PrecEn["<<i<<"] = "<<queriesPrecision[i]<<"\tRecallEn["<<i<<"] = "<<queriesRecall[i]<<"\t";
                                    out<<"PrecFr["<<i<<"] = "<<queriesPrecisionFr[i]<<"\tRecallFr["<<i<<"] = "<<queriesRecallFr[i]<<endl;
                                }

                                out<<"C1: "<< c1<<"\nC2: "<<c2<<endl;
                                out<<"numOfShownNonRel: "<<numOfShownNonRel<<"\nnumOfnotShownDoc: "<<numOfnotShownDoc<<endl;

                                //out<<"C1: "<< c1Fr<<"\nC2: "<<c2Fr<<endl;
                                //out<<"numOfShownNonRel: "<<numOfShownNonRelFr<<"\nnumOfnotShownDoc: "<<numOfnotShownDocFr<<endl;

                                double AVGP = avgPrecAll;//totalRelRec / totalRec;//(avgPrec+avgPrecFr)/2.0;
                                double AVGR = avgRecallAll;//totalRelRec / totalRel;//(avgRecall+avgRecallFr)/2.0;
                                out<<"Avg Precision: "<<AVGP<<"\t"<<avgPrec<<"\t"<<avgPrecFr<<endl;
                                out<<"Avg Recall: "<<AVGR<<"\t"<<avgRecall<<"\t"<<avgRecallFr<<endl;
                                out<<"F-measure: "<<(2*AVGP*AVGR)/(AVGP+AVGR)<<"\t"<<(2*avgPrec*avgRecall)/(avgPrec+avgRecall)<<"\t"<<(2*avgPrecFr*avgRecallFr)/(avgPrecFr+avgRecallFr) <<endl<<endl;



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
    ifstream infile;
    infile.open ("dictionary_fr2en_rahimi");
    string line;

    //bye , khodi # 0.4 , haha # 0.3 , aba 0.5
    //hi , salam goli # 0.9 , chetory # 0.1

    while (getline(infile,line))
    {
        string word, temp, tr;
        double prob;

        vector<pair<string, double> > trans;
        stringstream ss(line);

        ss >> word;
        ss>> temp;//comma

        while(ss >> temp)
        {
            if( temp != "#")
            {

                tr += temp+" ";
            }
            else
            {
                ss>> prob;

                string aaa = tr.substr(0,tr.size()-1);//remove last space
                std::transform(aaa.begin(), aaa.end(), aaa.begin(), ::tolower);//chon tuye index tolower hastan
                trans.push_back(make_pair<string, double>(aaa, prob));
                //cerr<<aaa<<" "<<prob<<endl;
                tr.clear();
                ss>>temp;//","
            }
        }

        vector<pair<string, double> > upperTrans;
        double totProb = 0;
        for(int i = 0 ; i < trans.size(); i++)
            totProb += trans[i].second;
        //cerr<<word<<" : ";
        for(int i = 0 ; i < trans.size(); i++)
        {
            trans[i].second /= totProb;
            if( trans[i].second > 0.1)
            {
                upperTrans.push_back(make_pair<string, double>(trans[i].first, trans[i].second) );
                //cerr<<trans[i].first<<" "<< trans[i].second<<" , ";

                //inv
                //map <string,vector<pair<string, double> > >dictionary,invDictionary;
                invDictionaryFr2En[trans[i].first].push_back(make_pair<string,double>(word,trans[i].second));


            }
            //cerr<<endl;
        }

        dictionaryFr2En.insert(make_pair<string, vector<pair<string, double> > >(word,upperTrans) );
    }
    infile.close();

    /******************************************************************/
    infile.open ("dictionary_en2fr_rahimi");

    while (getline(infile,line))
    {
        string word, temp, tr;
        double prob;

        vector<pair<string, double> > trans;
        stringstream ss(line);

        ss >> word;
        ss>> temp;//comma

        while(ss >> temp)
        {
            if( temp != "#")
            {

                tr += temp+" ";
            }
            else
            {
                ss>> prob;

                string aaa = tr.substr(0,tr.size()-1);//remove last space
                std::transform(aaa.begin(), aaa.end(), aaa.begin(), ::tolower);//chon tuye index tolower hastan
                trans.push_back(make_pair<string, double>(aaa, prob));
                //cerr<<aaa<<" "<<prob<<endl;
                tr.clear();
                ss>>temp;//","
            }
        }

        vector<pair<string, double> > upperTrans;
        double totProb = 0;
        for(int i = 0 ; i < trans.size(); i++)
            totProb += trans[i].second;
        //cerr<<word<<" : ";
        for(int i = 0 ; i < trans.size(); i++)
        {
            trans[i].second /= totProb;
            if( trans[i].second > 0.1)
            {
                upperTrans.push_back(make_pair<string, double>(trans[i].first, trans[i].second) );
                //cerr<<trans[i].first<<" "<< trans[i].second<<" , ";

                //inv
                //map <string,vector<pair<string, double> > >dictionary,invDictionary;
                invDictionaryEn2Fr[trans[i].first].push_back(make_pair<string,double>(word,trans[i].second));


            }
            //cerr<<endl;
        }

        dictionaryEn2Fr.insert(make_pair<string, vector<pair<string, double> > >(word,upperTrans) );
    }
    infile.close();

    /******************************************************************/

    /*cerr<<"##\n";
    map <string,vector<pair<string, double> > >::iterator it;
    for(it = invDictionaryFr2En.begin(); it != invDictionaryFr2En.end() ; ++it)
    {
        cerr<<it->first<<" : ";
        for(int i = 0 ; i < it->second.size(); i++)
        {
            cerr<< it->second[i].first<<"  "<<it->second[i].second<<" , ";
        }
        cerr<<endl;
    }*/

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
void writeIndexDocs2File(Index *ind)
{
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
                    doc[poses[i]] = ind->term(ti->termID());
                }
                //delete poses;
                //delete ti;
            }

            //outfile<<"<DOC>\n<DOCNO>"<<ind->document(docID)<<"</DOCNO>\n<TEXT>\n";
            for(int i = 0 ;i < doc.size();i++)
            {
                if(doc[i] != " ")
                    outfile<<doc[i]<<" ";
            }
            //outfile<<endl<<endl<<"</TEXT>\n</DOC>";
            outfile<<endl;

            //delete docTermInfoList;
        }

    }
    outfile.close();
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
void readWordEmbeddingFile(Index *ind,Index *otherInd)
{
    //int cc=0;
    cout << "ReadWordEmbeddingFile\n";
    string line,otherLine;

    ifstream in,otherIn;
#if 1
    if(WHO == 0)
    {
        if(DATASET == 0)
        {
            //in.open("/home/iis/Desktop/Edu/thesis/wordEmbeddingVector/infile_docs_Stemmed_withoutSW_W2V.vectors");
            //in.open("/home/iis/Desktop/RS-Framework/QE/QE/infile_docs_Stemmed_withoutSW_W2V.vectors");//server 69
            in.open("/home/iis/Desktop/Edu/thesis/wordEmbeddingVector/vectors_new_en_notStemmed.txt");
            otherIn.open("/home/iis/Desktop/Edu/thesis/wordEmbeddingVector/vectors_new_fr_notStemmed.txt");
        }

    }else if(WHO == 6)
    {
        if(DATASET == 0)//infile
        {
            in.open("/home/hosein/wordEmbeddingVector/vectors_new_en_notStemmed.txt");
            otherIn.open("/home/hosein/wordEmbeddingVector/vectors_new_fr_notStemmed.txt");
        }
    }
    else
    {
        if(DATASET == 0)//infile
        {
            in.open("/home/hossein/Desktop/IIS/Lemur/DataSets/wordEmbeddingVector/vectors_new_en_notStemmed.txt");
            otherIn.open("/home/hossein/Desktop/IIS/Lemur/DataSets/wordEmbeddingVector/vectors_new_fr_notStemmed.txt");
        }
    }
    getline(in,line);//first line is statistical in W2V
    getline(otherIn,otherLine);
#endif
#if 0
    ifstream in("/home/hossein/Desktop/IIS/Lemur/DataSets/wordEmbeddingVector/infile_vectors_100D_Glove.txt");
#endif
    //int cc=0;
    while(getline(in,line))
    {
        //cc++;
        istringstream iss(line);

        string sub;
        double dd;
        iss >> sub;

        /*if(sub.size() <= 1)
            {
                cerr<<sub<<" % ";
                continue;
            }*/

        int termID = ind->term(sub);

        while (iss>>dd)
            wordEmbedding[termID].push_back(dd);
    }
    //cerr<<cc<<"aaa\n\n\n";
    //cc=0;

    while(getline(otherIn,otherLine))
    {
        //cc++;
        istringstream iss(otherLine);

        string sub;
        double dd;
        iss >> sub;

        /*if(sub.size() <= 1)
            {
                cerr<<sub<<" % ";
                continue;
            }*/

        int termID = otherInd->term(sub);

        while (iss>>dd)
            wordEmbeddingFr[termID].push_back(dd);
    }
    //cerr<<cc<<"bbb\n\n\n";


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

void computeQueryAvgVec(Document *d, RetMethod *myMethod, bool isFr )
{
    cerr<<"WE size: "<<wordEmbedding.size()<<" WEfr size: "<<wordEmbeddingFr.size()<<endl;
    cerr<<"qTermIdVec size: "<<queryTermsIdVec.size()<<" qTermIdVecFr size: "<<queryTermsIdVecFr.size()<<endl;

    if(!isFr)
        queryTermsIdVec.clear();
    else
        queryTermsIdVecFr.clear();


    TextQuery *q = new TextQuery(*d);
    QueryRep *qr = myMethod->computeQueryRep(*q);
    TextQueryRep *textQR = (TextQueryRep *)(qr);


    std::map<int,vector<double> >::const_iterator endIt;
    if(!isFr)
        endIt = wordEmbedding.end();
    else
        endIt = wordEmbeddingFr.end();

    textQR->startIteration();
    while(textQR->hasMore())
    {
        QueryTerm *qt = textQR->nextTerm();
        std::map<int,vector<double> >::const_iterator it;
        if(!isFr)
            it = wordEmbedding.find(qt->id());
        else
            it = wordEmbeddingFr.find(qt->id());

        if(it != endIt)//found
        {
            if(!isFr)
            {
                for(int i=0; i < qt->weight() ; i++)   //(<queryW1,<1,2,4>)(queryW1,<1,2,3>)
                    queryTermsIdVec.push_back(make_pair<int , vector<double> > (qt->id() ,it->second ) );
            }else
            {
                for(int i=0; i < qt->weight() ; i++)   //(<queryW1,<1,2,4>)(queryW1,<1,2,3>)
                    queryTermsIdVecFr.push_back(make_pair<int , vector<double> > (qt->id() ,it->second ) );
            }
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

    if(!isFr)
    {

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
    else //isFr
    {
        vector<double> queryAvg( myMethod->W2VecDimSize ,0.0);
        for(int i = 0 ; i< queryTermsIdVecFr.size() ; i++)
        {
            for(int j = 0 ; j < myMethod->W2VecDimSize; j++)
                queryAvg[j] += queryTermsIdVecFr[i].second[j];
        }

        double qsize = queryTermsIdVecFr.size();
        for(int i = 0 ; i < myMethod->W2VecDimSize/*queryAvg.size()*/ ;i++)
            queryAvg[i] /= qsize;

        myMethod->Vq.clear();
        myMethod->Vq.assign( queryAvg.begin(),queryAvg.end());



        //weighted Query //dist from Avg
        weightedQueryTermsFr.clear();
        double totalSc = 0.0;
        for(int i = 0 ; i < queryTermsIdVecFr.size() ; i++)
        {
            //double sc = myMethod->softMaxFunc2(queryTermsIdVec[i].second , myMethod->Vq);
            double sc = myMethod->cosineSim(queryTermsIdVecFr[i].second , myMethod->Vq);

            weightedQueryTermsFr.push_back(make_pair<int, double>(queryTermsIdVecFr[i].first , sc));
            totalSc +=sc;
        }
        for(int i = 0 ;i< weightedQueryTermsFr.size() ;i++)
            weightedQueryTermsFr[i].second /= totalSc;

    }


    cerr<<"End WE size: "<<wordEmbedding.size()<<" WEfr size: "<<wordEmbeddingFr.size()<<endl;
    cerr<<"End qTermIdVec size: "<<queryTermsIdVec.size()<<" qTermIdVecFr size: "<<queryTermsIdVecFr.size()<<endl;


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

#endif

