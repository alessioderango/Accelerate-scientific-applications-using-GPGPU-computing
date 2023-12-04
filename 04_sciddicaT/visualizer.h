#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <regex>
#include <climits>
#include <chrono>
#include <thread>


using namespace std;

class Line{

    public:
    int x1,y1;
    int x2,y2;
    
    Line(int x1,int y1, int x2, int y2): x1(x1), y1(y1), x2(x2), y2(y2){

    }

    Line(){

    }

};

//ALLEGRO

BITMAP * buffer;
int pixelsQuadrato;

void initAllegro(){
	allegro_init();
	set_color_depth(32);
	install_keyboard();
	install_mouse();
}

void setAllegroDimension(int _bSchermo, int _hSchermo){

	set_gfx_mode(GFX_AUTODETECT_WINDOWED,_bSchermo, _hSchermo,0,0);
	show_mouse(screen);
	buffer = create_bitmap(_bSchermo,_hSchermo);

}

///////////////////////////////



int *maxStepVisited;
unordered_map<int,long int> *hashMap;
vector<int> totalNumberofSteps;
Line * lines;

long int gotoStep(int step,FILE * fp, int node){
    // return hashMap[node][step];
    // cout << step << endl;
    return hashMap[node][totalNumberofSteps[step]];
}


char * giveMeFileName(char* fileName,int node){
    char* fileNameTmp = new char[256];
    strcpy(fileNameTmp, fileName);
    strcat(fileNameTmp,std::to_string(node).c_str());
    strcat(fileNameTmp,".txt");
    return fileNameTmp;
}

char * giveMeFileNameIndex(char* fileName,int node){
    char* fileNameTmp = new char[256];
    strcpy(fileNameTmp, fileName);
    strcat(fileNameTmp,std::to_string(node).c_str());
    strcat(fileNameTmp,"_index.txt");
    return fileNameTmp;
}

FILE* giveMeLocalColAndRowFromStep(int step, char* fileName,int node, int &nLocalCols, int & nLocalRows, char*& line, size_t& len){

    char* fileNameTmp=giveMeFileName(fileName,node);
    //cout << " fileNameTmp "<< fileNameTmp << endl;

    FILE* fp = fopen(fileNameTmp, "r");
    if (fp == NULL){
        cout << "Can't read " << fileNameTmp << " in giveMeLocalColAndRowFromStep function" << endl;
        exit(EXIT_FAILURE);
    }
        
    long int fPos = gotoStep(step,fp,node);

    fseek(fp, fPos, SEEK_SET);

    //printMatrixFromStepByUser(hashmap, stepUser);	
    getline(&line, &len, fp);
    // cout << line << endl;
    char * pch;
    pch = strtok (line,"-");
    nLocalCols =atoi(pch);
    pch = strtok (NULL, "-");
    nLocalRows =atoi(pch);
    
    return fp;
}


template <class T>
void getElementMatrix(int step, T**& m,int nGlobalCols, int nGlobalRows,int nNodeX, int nNodeY,char* fileName, Line* lines ){
    

    int * AlllocalCols = new int[(nNodeX*nNodeY)];
    int * AlllocalRows = new int[(nNodeX*nNodeY)];

    // m = new T*[nGlobalRows];
    // for(int i = 0;i < nGlobalRows; i++){
    //     m[i]= new T[nGlobalCols];
    // }
  
  
    for(int node = 0; node < (nNodeX*nNodeY);node++)
    {
       	int nLocalCols;
        int nLocalRows;
        char* line = NULL;
        size_t len = 0;
        
        FILE* fp = giveMeLocalColAndRowFromStep(step, fileName, node, nLocalCols, nLocalRows , line, len);

        AlllocalCols[node] = nLocalCols;
        AlllocalRows[node] = nLocalRows;
        fclose(fp);
    }
    //for(int k = 0; k < (nNodeX*nNodeY); k++)
    //{
    //        cout << "AlllocalCols[ " << k << "] " << AlllocalCols[k]<< endl;
    //}
    //    
    //for(int k = 0; k < (nNodeX*nNodeY); k++)
    //{
    //         cout << "AlllocalRows[ " << k << "] " << AlllocalRows[k]<< endl;
    //}

    bool startStepDone = false;
    for(int node = 0; node < (nNodeX*nNodeY);node++)
    {
        int nLocalCols;
        int nLocalRows;
        char* line = NULL;
        size_t len = 0;

        FILE* fp = giveMeLocalColAndRowFromStep(step, fileName, node, nLocalCols, nLocalRows , line, len);
        
        int offsetX =0;//= //(node % nNodeX)*nLocalCols;//-this->borderSizeX;
        int offsetY =0;//= //(node / nNodeX)*nLocalRows;//-this->borderSizeY;

        //cout << endl;
        //cout << endl;

        //cout << "Node " << node<< endl;
        //cout << "nNodeX " << nNodeX<< endl;
        //cout << "nNodeY " << nNodeY<< endl;
        //cout << "nLocalCols=" << nLocalCols << " nLocalRows=" << nLocalRows << endl;


        if (nNodeY == 1)
        {
            for (int k = 0; k < node % nNodeX; k++)
            {
                offsetX += AlllocalCols[k];
            }
        }
        else
        {
            for (int k = (node / nNodeX) * nNodeX; k < node; k++)
            {
                offsetX += AlllocalCols[k];
            }
        }

        if (node >= nNodeX)
        {
            for (int k = node - nNodeX; k >= 0;)
            {
                offsetY += AlllocalRows[k];
                k -= nNodeX;
            }
        }
        //cout << "offsetX=" << offsetX << " offsetY=" << offsetY << endl;
        //cout << endl;
        //cout << endl;

        

        Line *lineTmp= new Line(offsetX, offsetY,offsetX+nLocalCols, offsetY);
        Line *lineTmp2= new Line(offsetX, offsetY,offsetX, offsetY+nLocalRows);

        lines[node*2]   = *lineTmp;
        lines[node*2+1] = *lineTmp2;
        
        //cout << " lineTmp.x1= " << lineTmp->x1 << " lineTmp.y1 " << lineTmp->y1 << " lineTmp.x2= " << lineTmp->x2 << " lineTmp.y2 " << lineTmp->y2<< endl;
        //cout << " lineTmp2.x1= " << lineTmp2->x1 << " lineTmp2.y1 " << lineTmp2->y1 << " lineTmp2.x2= " << lineTmp2->x2 << " lineTmp2.y2 " << lineTmp2->y2<< endl;

        int row=0;
       
        while (row < nLocalRows){
            getline(&line, &len, fp);
            int col=0;
            //m[row]=new T[nLocalCols];
            while (col < nLocalCols){
                char* elem;
                if (col==0)
                    elem = strtok(line," ");
                else    
                    elem = strtok(NULL," ");
                // cout << elem << endl;
                
                if (!startStepDone){
                    m[row+offsetY][col+offsetX].T::startStep(step);
                    startStepDone=true;
                }
                // cout << endl;
                // cout << node << " elem = " <<elem << endl;
                // cout << node << " " <<nGlobalRows << " " << nGlobalCols << endl;
                // cout << node << " " <<nLocalRows << " " << nLocalCols << endl;
                // cout << node << " " <<row << " " << col << endl;
                // cout << node << " " <<row+offsetY << " " << col+offsetX << endl;
                // cout << endl;

                m[row+offsetY][col+offsetX].T::composeElement(elem);

                //rgb* color= m[row+offsetY][col+offsetX].outputValue();
             //   cout << color->getRed()<<" " << color->getGreen()<<" " <<color->getBlue() << endl;

                col++;
            }
            row++;
        }

        fclose(fp);

    }

  
    delete []AlllocalCols;
    delete []AlllocalRows;
    




}

template <class T>
void drawWithAllegro(T** p,int nRows,int nCols, int step, Line * lines, int dimLines, string edittext){

        
        for(int row=0;row<nRows;row++){
            for(int col=0;col<nCols;col++){                  
                
                rgb* color= p[row][col].outputValue();
                //cout << color->getRed()<<" " << color->getGreen()<<" " <<color->getBlue() << endl;

                rectfill(buffer,col*pixelsQuadrato, row*pixelsQuadrato, col*pixelsQuadrato+pixelsQuadrato,row*pixelsQuadrato+pixelsQuadrato, makecol(color->getRed(), color->getGreen(),color->getBlue()));

                //delete color;
                //putpixel(buffer,row,col, makecol(255,0,0));//color->getRed(), color->getGreen(),color->getBlue()));
            }    
        }

        for(int i =0; i < dimLines; i++)
        {
            //cout << lines[i].x1 << "  " << lines[i].y1 << "  " <<lines[i].x2 << "  " <<lines[i].y2 << endl;
            line(buffer,lines[i].x1*pixelsQuadrato, lines[i].y1*pixelsQuadrato,lines[i].x2*pixelsQuadrato,lines[i].y2*pixelsQuadrato, makecol(0,0,0));
        }

        textprintf_ex(buffer,font , 300,0, makecol(255,255,255), makecol(0,0,0), "step %d ", totalNumberofSteps[step]);
        blit(buffer, screen, 0,0,0,0, nCols*pixelsQuadrato, (nRows+10)*pixelsQuadrato);
}

void readConfigurationFile( char*filename, int infoFromFile[8], char* outputFileName){
	char str[999];
	int n =0;
	FILE * file;
	file = fopen(filename, "r");
    
	if (file) {
		int i =1;
		while (fscanf(file, "%s", str)!=EOF && i <= 16){
			if(i%2 == 0)
				infoFromFile[n++]=atoi(str);	
			++i;
		}

        int x = fscanf(file, "%s", str);
        int y = fscanf(file, "%s", str);
        strcpy(outputFileName, str);
        
		fclose(file);
	}
}

void loadHashmapFromFile(int nNodeX, int nNodeY, char*filename){
    for(int node = 0; node < (nNodeX*nNodeY);node++)
    {
        char * fileNameIndex =giveMeFileNameIndex(filename,node);   
        cout << fileNameIndex << endl;   
        FILE* fp  = fopen(fileNameIndex, "r");
        if (fp == NULL)
        {
          cout << "Canot open " << fileNameIndex << endl;
            exit(EXIT_FAILURE);
        }
        int step =0;
        long int nbytes = 0;
        while( fscanf(fp, "%d %ld\n", &step, &nbytes) != EOF){
               // cout << step << " " << nbytes << endl;
                 std::pair<int, long int> p(step,nbytes);
                 //cout <<  " inserisco : " << step << endl;
                 if(node == 0)
                    totalNumberofSteps.push_back(step);
                 hashMap[node].insert(p);
        }
        // char* line = NULL;
	    // size_t len = 0;
        // while( getline(&line, &len, fp)){                
        //     char * pch;
        //     pch = strtok (line," ");
        //     int step =atoi(pch);
        //     pch = strtok (NULL, " ");
        //     long int nbytes =strtoll(pch, NULL, 10);//atoi(pch);

        //     cout << step << " " << nbytes << endl;
        //     std::pair<int, long int> p(step,nbytes);
        //     hashMap[node].insert(p);
        //     cout << fileNameIndex << endl;   
        // }
        // cout << "finito" << endl;
        fclose(fp);
        //cout << "finito" << endl;      
    }
}

template <class T>
int visualizer(int argc, char *argv[]) {

    if (argc == 1)
    {
        cout << " no pixel size " << endl;
        return 0;
    }
    char *filename = argv[1];
    pixelsQuadrato = atoi(argv[2]);
    int infoFromFile[8];
    char* outputFileName = new char[256];
    string tmpString = filename;
    std::size_t found = tmpString.find_last_of("/\\");
    tmpString = tmpString.substr(0,found);
    tmpString+= "/Output/";
    char * firstS  = new char[256];
    strcpy (firstS, tmpString.c_str());

    readConfigurationFile(filename, infoFromFile, outputFileName);

    strcat(firstS,outputFileName);
    strcpy(outputFileName,firstS);

	  int dimX = infoFromFile[0];//numero colonne
	  int dimY = infoFromFile[1];//numero righe
	  int borderSizeX = infoFromFile[2]; 
	  int borderSizeY = infoFromFile[3]; 
	  int numBorders = infoFromFile[4];
	  int nNodeX = infoFromFile[5]; 
	  int nNodeY = infoFromFile[6]; 
	  int nsteps = infoFromFile[7]; 

    cout << "dimX " << dimX << endl;
    cout << "dimY " << dimY << endl;
    cout << "borderSizeX " << borderSizeX << endl;
    cout << "borderSizeY " << borderSizeY << endl;
    cout << "numBorders " << numBorders << endl;
    cout << "nNodeX " << nNodeX << endl;
    cout << "nNodeY " << nNodeY << endl;
    cout << "outputFileName " << outputFileName << endl;

    hashMap = new unordered_map<int,long int>[nNodeX*nNodeY];
    maxStepVisited = new int[nNodeX*nNodeY];
    int numberOfLines = 2*(nNodeX*nNodeY);
    char c='s';

    initAllegro();
    int step=1;
    //while(c!='n'){
    bool changed = false;
    bool firstTime = true;
    bool insertAction = false;
    string  edittext;                         // an empty string for editting
    string::iterator iter = edittext.begin(); // string iterator
    int     caret  = 0;                       // tracks the text caret
    bool    insert = true;                    // true if text should be inserted
    T** p;

    p = new T*[dimY];
    for(int i = 0;i < dimY; i++){
        p[i]= new T[dimX];
    }
    //cout << dimX << " " << dimY << endl;
    
    setAllegroDimension(dimX*pixelsQuadrato,(dimY+10)*pixelsQuadrato );

    loadHashmapFromFile(nNodeX,nNodeY,outputFileName);
      
    while(!key[KEY_ESC]){
        
        if(key[KEY_UP])
        {
            if(step < hashMap[0].size())
            step+=1;
            changed = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
       
        if(key[KEY_DOWN])
        {
            if(step > 1)
            step-=1;
            changed = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

        }
        if(key[KEY_I])
		{
            insertAction = true;

        }
    while(keypressed() && insertAction)
      {
         int  newkey   = readkey();
        
         char ASCII    = newkey & 0xff;
         char scancode = newkey >> 8;
         
        
         // a character key was pressed; add it to the string
         if(ASCII >= 32 && ASCII <= 126)
         {
            // add the new char, inserting or replacing as need be
            if(insert || iter == edittext.end())
               iter = edittext.insert(iter, ASCII);
            else
               edittext.replace(caret, 1, 1, ASCII);

            // increment both the caret and the iterator
            caret++;
            iter++;
         }
         // some other, "special" key was pressed; handle it here
         else
            switch(scancode)
            {
               case KEY_DEL:
                  if(iter != edittext.end()) iter = edittext.erase(iter);
               break;

               case KEY_BACKSPACE:
                  if(iter != edittext.begin())
                  {
                     caret--;
                     iter--;
                     iter = edittext.erase(iter);
                  }
               break;
            
               case KEY_RIGHT:
                  if(iter != edittext.end())   caret++, iter++;
               break;
            
               case KEY_LEFT:
                  if(iter != edittext.begin()) caret--, iter--;
               break;
            
               case KEY_INSERT:
                  insert = !insert;
               break;

               case KEY_ENTER:
                    step = atoi(edittext.c_str());
                    changed = true;
                    insertAction = false;
                    
               break;

               default:

               break;
            }
                        
            clear(buffer);
            textprintf_ex(buffer, font, 0, (dimY+5)*pixelsQuadrato, makecol(255,255,255),makecol(0,0,0), "jump to %s",edittext.c_str());
            lines = new Line[numberOfLines];
            getElementMatrix(step, p, dimX, dimY, nNodeX, nNodeY, outputFileName, lines);
           
            drawWithAllegro(p, dimY, dimX, step, lines, numberOfLines,edittext);
            //cout << step << endl;
            //step++;

            delete [] lines;    

      }


        if(changed || firstTime){
            
            lines = new Line[numberOfLines];
            getElementMatrix(step, p, dimX, dimY, nNodeX, nNodeY, outputFileName, lines);
           
            drawWithAllegro(p, dimY, dimX, step, lines, numberOfLines,edittext);
            firstTime = false;
            changed = false;
            //cout << step << endl;
            //step++;

            delete [] lines;    
           
        }
  
    }
     

    for(int i = 0;i < nNodeY; i++){
        delete p[i];
    }
    delete [] p;
    delete [] hashMap;
    delete [] maxStepVisited;
   
    return(0);
}

