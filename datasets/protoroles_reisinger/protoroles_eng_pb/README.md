# Semantic Proto-Roles

Data from the experiment described in Section 5 of:

Drew Reisinger, Rachel Rudinger, Francis Ferraro, Craig Harman, Kyle Rawlins, and Benjamin Van Durme. 2015. Semantic Proto-Roles.
Transactions of the Association for Computational Linguistics, vol 3, pp 475 - 488.

[link](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/674)

## Column descriptions

| Column      | Description |
|-------------|-------------|
| Split       | The training split used in training the model described in Sec. 7 of the paper; either `train`, `dev`, or `test` |
| Sentence.ID | The file and sentence number of the sentence in the Penn Treebank in the format `FILENUM_SENTNUM`, e.g. `1778_51` means sentence 51 of wsj/17/wsj_1778.mrg |
| Pred.Token  | The position of the predicate in the sentence starting at zero |
| Arg         | The PropBank argument label; either `0`, `1`, `2`, `3`, `4`, or `5` |
| Arg.Pos     | The position of the argument in the syntactic tree; for an explanation of syntax, see [PropBank data format](http://verbs.colorado.edu/~mpalmer/projects/ace/EPB-data-format.txt) |
| Roleset     | The PropBank roleset of the predicate |
| Gram.Func   | The grammatical function of the argument; either `subj`, `obj` (for direct object), or `other` |
| Property    | The proto-role property being annotated |
| Response    | The 5-point Likert scale annotation of the property, where `1` means _very unlikely_ and `5` means _very likely_ |
| Applicable  | Whether the property is applicable in principle to the argument in question; either `True` or `False` |
