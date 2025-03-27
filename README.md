# veridion_intern
Veridion Challenge
În acest proiect, mi-am propus să dezvolt un clasificator automatizat capabil să atribuie etichete relevante companiilor dintr-o taxonomie specifică industriei de asigurări, utilizând informații precum descrierea companiei și alte caracteristici disponibile. Abordarea mea a fost structurată pentru a asigura claritate, transparență și eficiență, detaliind fiecare etapă a procesului, de la preprocesarea datelor până la validare și optimizare. Mai jos explic obiectivele, metodologia, motivația alegerilor făcute, avantajele soluției, limitările și raționamentul din spatele deciziilor.
Obiective
•	Date de intrare: O listă de companii cu informații precum Company Description, Business Tags, Sector, Category și Niche.
•	Taxonomie fixă: O listă predefinită de etichete specifice industriei de asigurări.
•	Clasificator eficient: Atribuirea corectă a companiilor la una sau mai multe etichete relevante.
•	Validare și analiză: Evaluarea eficacității soluției și identificarea punctelor forte și a aspectelor care necesită îmbunătățiri.
Abordarea mea
Am construit soluția pas cu pas, punând accent pe acuratețe, scalabilitate și interpretare semantică profundă. Iată cum am procedat:
1.	Preprocesarea datelor
Am combinat câmpurile relevante (descriere, categorie, nișă) într-un text unificat, eliminând caracterele speciale și standardizând formatul pentru o analiză mai clară. De asemenea, am exclus companiile fără business tags pentru a evita clasificările incerte, reducând astfel riscul de erori.
2.	Generarea embeddings-urilor semantice
Am utilizat modelul SentenceTransformer (all-MiniLM-L6-v2) pentru a transforma datele textuale ale companiilor și taxonomia într-o formă numerică optimizată semantic. Această metodă capturează mai bine contextul și sensul descrierilor comparativ cu abordări mai simple, precum TF-IDF.
3.	Clasificare prin similaritate cosinus
Am calculat similaritatea cosinus între embedding-urile companiilor și cele ale etichetelor din taxonomie pentru a identifica cele mai relevante categorii. Fiecare companie a primit până la trei etichete, aplicând un prag minim de încredere (min_confidence=0.6), care echilibrează precizia și acoperirea. Rezultatele au fost salvate într-un fișier optimized_classified_companies.csv, sortate după gradul de încredere.
4.	Filtrare și validare
Am verificat relevanța etichetelor atribuite prin compararea lor cu business tags și categoria generală a fiecărei companii, excluzând cazurile unde nu exista o potrivire clară. Pentru a evalua acuratețea, am analizat manual un eșantion de date, ajustând soluția pe baza observațiilor.
5.	Optimizare și scalabilitate
Pentru a gestiona volume mari de date, am crescut batch_size și am integrat procesare paralelă (prin joblib.Parallel) pentru a accelera atribuirea etichetelor. Am adăugat suport pentru Spark/Dask și cuantizare, optimizând performanța, și am implementat salvarea incrementală a rezultatelor pentru eficiență.
Motivația alegerii acestei soluții
Am optat pentru embeddings semantice deoarece oferă o interpretare profundă a textului, superioară metodelor bazate pe frecvență, cum ar fi TF-IDF. Filtrarea inteligentă bazată pe business tags și pragul de încredere minimizează atribuirile incorecte, iar arhitectura soluției permite scalabilitate pentru seturi mari de date. Pragul de 0.6 a fost ales empiric pentru a optimiza echilibrul între precizie și acoperire, evitând atât clasificările excesiv de restrictive, cât și pe cele excesiv de generoase.
Avantaje și aspecte de îmbunătățit
Puncte forte:
•	Capacitate avansată de interpretare semantică, esențială pentru descrieri complexe.
•	Scalabilitate ridicată, potrivită pentru procesarea unor volume mari de date.
•	Reducerea erorilor prin filtre eficiente și validare manuală.
Limitări:
•	Excluderea companiilor fără business tags poate duce la pierderea unor informații utile.
•	Modelul funcționează optim doar în limba engleză, limitând aplicabilitatea la alte limbi.
•	Lipsa unui set de date de referință face evaluarea performanței mai subiectivă.
Raționamentul din spatele soluției
Obiectivul principal a fost să creez un clasificator robust, capabil să înțeleagă contextul descrierilor companiilor și să atribuie etichete relevante din taxonomie. Alegerea embeddings-urilor semantice și a similarității cosinus a fost motivată de nevoia de a captura nuanțe care ar fi fost pierdute cu metode mai simple. Preprocesarea riguroasă și filtrarea au asigurat calitatea datelor, iar optimizările precum procesarea paralelă și cuantizarea au garantat scalabilitatea. Validarea manuală a fost un pas esențial pentru a confirma acuratețea și a ajusta soluția, compensând absența unui set de date de referință.
Această metodologie combină precizia semantică cu eficiența computațională, oferind o soluție echilibrată și adaptabilă, potrivită pentru clasificarea automată a companiilor în industria asigurărilor.

