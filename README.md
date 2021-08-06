# Course_Deep_Learning_Pytorch_Tensorflow


### To-Do
* TopK-Berechnung überdenken (Anmerkung Lisa)
  * 1 - topk_percentage, um tatsächlich die die oberen x % und nicht (1 - x)% zu erwischen bei der Maske
  * Quantilsberechnung auf Absolutwerte umstellen (geht ja um magnitude ohne Vorzeichen)
* Gradients truly sparse machen
  * Aktuell: compute as dense, mask, turn to sparse $\Rightarrow$ leider nicht wirklich sparse
  * Sven irgendwann: 
    * Irgendwie in grad_weight = grad_output.t().mm(input) bereits in grad_output/input sparsity zu erzeugen, sodass man da nicht dense Matrizen multiplizieren muss
    * Funktioniert, soweit ich es jetzt verstehe, nicht. Der Gradient ist bei Batch Size = 1, das outer product von zwei Vektoren und wenn man einzelne Werte in den beiden Vektoren auf 0 setzt kann man nur ein input neuron oder ein output neuron und die damit einhergehenden weights auf 0 setzen. D.h. man kann nicht einzelne weights auf 0 setzen, sondern nur auf Neuronebene arbeiten. Wenn man also einzelne Weights auf 0 setzen will, dann muss man das post ausrechnen der Gradienten machen
* Loss-Funktion implementieren
* In Routine packen, wobei
  * nicht unbedingt in jedem Step das active set angepasst wird (Paper: z. B. alle 100)
  * nach einer exploration phase mit Anpassungen des active set eine refinement phase mit fixen aktiven Gewichten kommt

### Thoughts & Ideas
* Wir sollten uns einen fancy date aggregator schreiben, damit wir verschiedene loss KPI speichern und uns anschauen können. Sowas wie Angle oder l2 Norm aller Parameter
* Bezieht sich sparsity eigentlich auch auf Bias? Oder ist der wurscht, weil Dim klein im Vergleich zu weight matrices?
* if-else-Sachen im forward pass
  * Brauchen wir eine Implementierung für sparse = False wirklich?
  * Der Teil mit training = False ist ja praktisch der prediction step - ist es üblich, so was _im_ Layer zu spezifizieren?

### Proof of Concept
Erst mal nicht generisch, sondern minimale funktionierende Lösung -> Resnet50 wie in dem Paper beschrieben bietet sich an. Zuallererst aber mal kleine Schritte ausprobieren mit torch.sparse

### Ideas for Testing
Subroutinen für: assert effective dimensionality?
Früher oder später Modul-Ordnerstruktur mit eigenem Testordner?
