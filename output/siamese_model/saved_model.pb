??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@0*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:0*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@0*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@0*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:0*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?/
value?/B?/ B?/
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
 
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?
 iter

!beta_1

"beta_2
	#decay
$learning_ratemumv%mw&mx'my(mzv{v|%v}&v~'v(v?
*
%0
&1
'2
(3
4
5
 
*
%0
&1
'2
(3
4
5
?
)metrics
	variables

*layers
+layer_regularization_losses
,layer_metrics
regularization_losses
	trainable_variables
-non_trainable_variables
 
 
h

%kernel
&bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
R
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
R
:	variables
;regularization_losses
<trainable_variables
=	keras_api
h

'kernel
(bias
>	variables
?regularization_losses
@trainable_variables
A	keras_api

%0
&1
'2
(3
 

%0
&1
'2
(3
?
Bmetrics
	variables

Clayers
Dlayer_regularization_losses
Elayer_metrics
regularization_losses
trainable_variables
Fnon_trainable_variables
 
 
 
?
Gmetrics
	variables

Hlayers
Ilayer_regularization_losses
Jlayer_metrics
regularization_losses
trainable_variables
Knon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Lmetrics
	variables

Mlayers
Nlayer_regularization_losses
Olayer_metrics
regularization_losses
trainable_variables
Pnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
#
0
1
2
3
4
 
 
 

%0
&1
 

%0
&1
?
Smetrics
.	variables

Tlayers
Ulayer_regularization_losses
Vlayer_metrics
/regularization_losses
0trainable_variables
Wnon_trainable_variables
 
 
 
?
Xmetrics
2	variables

Ylayers
Zlayer_regularization_losses
[layer_metrics
3regularization_losses
4trainable_variables
\non_trainable_variables
 
 
 
?
]metrics
6	variables

^layers
_layer_regularization_losses
`layer_metrics
7regularization_losses
8trainable_variables
anon_trainable_variables
 
 
 
?
bmetrics
:	variables

clayers
dlayer_regularization_losses
elayer_metrics
;regularization_losses
<trainable_variables
fnon_trainable_variables

'0
(1
 

'0
(1
?
gmetrics
>	variables

hlayers
ilayer_regularization_losses
jlayer_metrics
?regularization_losses
@trainable_variables
knon_trainable_variables
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ltotal
	mcount
n	variables
o	keras_api
D
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

n	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1

s	variables
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_45584
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_46084
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biastotalcounttotal_1count_1Adam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_46175Ņ
?T
?
 __inference__wrapped_model_45059
input_1
input_2O
5model_1_model_conv2d_1_conv2d_readvariableop_resource:@D
6model_1_model_conv2d_1_biasadd_readvariableop_resource:@D
2model_1_model_dense_matmul_readvariableop_resource:@0A
3model_1_model_dense_biasadd_readvariableop_resource:0@
.model_1_dense_1_matmul_readvariableop_resource:=
/model_1_dense_1_biasadd_readvariableop_resource:
identity??&model_1/dense_1/BiasAdd/ReadVariableOp?%model_1/dense_1/MatMul/ReadVariableOp?-model_1/model/conv2d_1/BiasAdd/ReadVariableOp?/model_1/model/conv2d_1/BiasAdd_1/ReadVariableOp?,model_1/model/conv2d_1/Conv2D/ReadVariableOp?.model_1/model/conv2d_1/Conv2D_1/ReadVariableOp?*model_1/model/dense/BiasAdd/ReadVariableOp?,model_1/model/dense/BiasAdd_1/ReadVariableOp?)model_1/model/dense/MatMul/ReadVariableOp?+model_1/model/dense/MatMul_1/ReadVariableOp?
,model_1/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_1_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,model_1/model/conv2d_1/Conv2D/ReadVariableOp?
model_1/model/conv2d_1/Conv2DConv2Dinput_14model_1/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/model/conv2d_1/Conv2D?
-model_1/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/model/conv2d_1/BiasAdd/ReadVariableOp?
model_1/model/conv2d_1/BiasAddBiasAdd&model_1/model/conv2d_1/Conv2D:output:05model_1/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
model_1/model/conv2d_1/BiasAdd?
model_1/model/conv2d_1/ReluRelu'model_1/model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/model/conv2d_1/Relu?
%model_1/model/max_pooling2d_1/MaxPoolMaxPool)model_1/model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2'
%model_1/model/max_pooling2d_1/MaxPool?
 model_1/model/dropout_1/IdentityIdentity.model_1/model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2"
 model_1/model/dropout_1/Identity?
=model_1/model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=model_1/model/global_average_pooling2d/Mean/reduction_indices?
+model_1/model/global_average_pooling2d/MeanMean)model_1/model/dropout_1/Identity:output:0Fmodel_1/model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2-
+model_1/model/global_average_pooling2d/Mean?
)model_1/model/dense/MatMul/ReadVariableOpReadVariableOp2model_1_model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02+
)model_1/model/dense/MatMul/ReadVariableOp?
model_1/model/dense/MatMulMatMul4model_1/model/global_average_pooling2d/Mean:output:01model_1/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model/dense/MatMul?
*model_1/model/dense/BiasAdd/ReadVariableOpReadVariableOp3model_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02,
*model_1/model/dense/BiasAdd/ReadVariableOp?
model_1/model/dense/BiasAddBiasAdd$model_1/model/dense/MatMul:product:02model_1/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model/dense/BiasAdd?
.model_1/model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp5model_1_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.model_1/model/conv2d_1/Conv2D_1/ReadVariableOp?
model_1/model/conv2d_1/Conv2D_1Conv2Dinput_26model_1/model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2!
model_1/model/conv2d_1/Conv2D_1?
/model_1/model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_1/model/conv2d_1/BiasAdd_1/ReadVariableOp?
 model_1/model/conv2d_1/BiasAdd_1BiasAdd(model_1/model/conv2d_1/Conv2D_1:output:07model_1/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 model_1/model/conv2d_1/BiasAdd_1?
model_1/model/conv2d_1/Relu_1Relu)model_1/model/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@2
model_1/model/conv2d_1/Relu_1?
'model_1/model/max_pooling2d_1/MaxPool_1MaxPool+model_1/model/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2)
'model_1/model/max_pooling2d_1/MaxPool_1?
"model_1/model/dropout_1/Identity_1Identity0model_1/model/max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@2$
"model_1/model/dropout_1/Identity_1?
?model_1/model/global_average_pooling2d/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2A
?model_1/model/global_average_pooling2d/Mean_1/reduction_indices?
-model_1/model/global_average_pooling2d/Mean_1Mean+model_1/model/dropout_1/Identity_1:output:0Hmodel_1/model/global_average_pooling2d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2/
-model_1/model/global_average_pooling2d/Mean_1?
+model_1/model/dense/MatMul_1/ReadVariableOpReadVariableOp2model_1_model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02-
+model_1/model/dense/MatMul_1/ReadVariableOp?
model_1/model/dense/MatMul_1MatMul6model_1/model/global_average_pooling2d/Mean_1:output:03model_1/model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model/dense/MatMul_1?
,model_1/model/dense/BiasAdd_1/ReadVariableOpReadVariableOp3model_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02.
,model_1/model/dense/BiasAdd_1/ReadVariableOp?
model_1/model/dense/BiasAdd_1BiasAdd&model_1/model/dense/MatMul_1:product:04model_1/model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model/dense/BiasAdd_1?
model_1/lambda/subSub$model_1/model/dense/BiasAdd:output:0&model_1/model/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????02
model_1/lambda/sub?
model_1/lambda/SquareSquaremodel_1/lambda/sub:z:0*
T0*'
_output_shapes
:?????????02
model_1/lambda/Square?
$model_1/lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$model_1/lambda/Sum/reduction_indices?
model_1/lambda/SumSummodel_1/lambda/Square:y:0-model_1/lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
model_1/lambda/Sumy
model_1/lambda/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model_1/lambda/Maximum/y?
model_1/lambda/MaximumMaximummodel_1/lambda/Sum:output:0!model_1/lambda/Maximum/y:output:0*
T0*'
_output_shapes
:?????????2
model_1/lambda/Maximumq
model_1/lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lambda/Const?
model_1/lambda/Maximum_1Maximummodel_1/lambda/Maximum:z:0model_1/lambda/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/lambda/Maximum_1?
model_1/lambda/SqrtSqrtmodel_1/lambda/Maximum_1:z:0*
T0*'
_output_shapes
:?????????2
model_1/lambda/Sqrt?
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_1/MatMul/ReadVariableOp?
model_1/dense_1/MatMulMatMulmodel_1/lambda/Sqrt:y:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_1/MatMul?
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOp?
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_1/BiasAdd?
model_1/dense_1/SigmoidSigmoid model_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_1/Sigmoidv
IdentityIdentitymodel_1/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp.^model_1/model/conv2d_1/BiasAdd/ReadVariableOp0^model_1/model/conv2d_1/BiasAdd_1/ReadVariableOp-^model_1/model/conv2d_1/Conv2D/ReadVariableOp/^model_1/model/conv2d_1/Conv2D_1/ReadVariableOp+^model_1/model/dense/BiasAdd/ReadVariableOp-^model_1/model/dense/BiasAdd_1/ReadVariableOp*^model_1/model/dense/MatMul/ReadVariableOp,^model_1/model/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2^
-model_1/model/conv2d_1/BiasAdd/ReadVariableOp-model_1/model/conv2d_1/BiasAdd/ReadVariableOp2b
/model_1/model/conv2d_1/BiasAdd_1/ReadVariableOp/model_1/model/conv2d_1/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv2d_1/Conv2D/ReadVariableOp,model_1/model/conv2d_1/Conv2D/ReadVariableOp2`
.model_1/model/conv2d_1/Conv2D_1/ReadVariableOp.model_1/model/conv2d_1/Conv2D_1/ReadVariableOp2X
*model_1/model/dense/BiasAdd/ReadVariableOp*model_1/model/dense/BiasAdd/ReadVariableOp2\
,model_1/model/dense/BiasAdd_1/ReadVariableOp,model_1/model/dense/BiasAdd_1/ReadVariableOp2V
)model_1/model/dense/MatMul/ReadVariableOp)model_1/model/dense/MatMul/ReadVariableOp2Z
+model_1/model/dense/MatMul_1/ReadVariableOp+model_1/model/dense/MatMul_1/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
@__inference_model_layer_call_and_return_conditional_losses_45757

inputsA
'conv2d_1_conv2d_readvariableop_resource:@6
(conv2d_1_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@03
%dense_biasadd_readvariableop_resource:0
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_1/Identity?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeandropout_1/Identity:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
global_average_pooling2d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/BiasAddq
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_45798

inputs!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_451662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_45166

inputs(
conv2d_1_45124:@
conv2d_1_45126:@
dense_45160:@0
dense_45162:0
identity?? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_45124conv2d_1_45126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_451232"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_451332!
max_pooling2d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_451402
dropout_1/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_451472*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_45160dense_45162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_451592
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity?
NoOpNoOp!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
R
&__inference_lambda_layer_call_fn_45845
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_453572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
?
?
@__inference_model_layer_call_and_return_conditional_losses_45303
input_3(
conv2d_1_45289:@
conv2d_1_45291:@
dense_45297:@0
dense_45299:0
identity?? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_1_45289conv2d_1_45291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_451232"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_451332!
max_pooling2d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_451402
dropout_1/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_451472*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_45297dense_45299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_451592
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity?
NoOpNoOp!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?

?
'__inference_model_1_layer_call_fn_45718
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_453772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?s
?
!__inference__traced_restore_46175
file_prefix1
assignvariableop_dense_1_kernel:-
assignvariableop_1_dense_1_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
"assignvariableop_7_conv2d_1_kernel:@.
 assignvariableop_8_conv2d_1_bias:@1
assignvariableop_9_dense_kernel:@0,
assignvariableop_10_dense_bias:0#
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: ;
)assignvariableop_15_adam_dense_1_kernel_m:5
'assignvariableop_16_adam_dense_1_bias_m:D
*assignvariableop_17_adam_conv2d_1_kernel_m:@6
(assignvariableop_18_adam_conv2d_1_bias_m:@9
'assignvariableop_19_adam_dense_kernel_m:@03
%assignvariableop_20_adam_dense_bias_m:0;
)assignvariableop_21_adam_dense_1_kernel_v:5
'assignvariableop_22_adam_dense_1_bias_v:D
*assignvariableop_23_adam_conv2d_1_kernel_v:@6
(assignvariableop_24_adam_conv2d_1_bias_v:@9
'assignvariableop_25_adam_dense_kernel_v:@03
%assignvariableop_26_adam_dense_bias_v:0
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_dense_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27f
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_28?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
T
8__inference_global_average_pooling2d_layer_call_fn_45960

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_451472
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_45916

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_45159

inputs0
matmul_readvariableop_resource:@0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?J
?
B__inference_model_1_layer_call_and_return_conditional_losses_45635
inputs_0
inputs_1G
-model_conv2d_1_conv2d_readvariableop_resource:@<
.model_conv2d_1_biasadd_readvariableop_resource:@<
*model_dense_matmul_readvariableop_resource:@09
+model_dense_biasadd_readvariableop_resource:08
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?'model/conv2d_1/BiasAdd_1/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?&model/conv2d_1/Conv2D_1/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?$model/dense/BiasAdd_1/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?#model/dense/MatMul_1/ReadVariableOp?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dinputs_0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/BiasAdd?
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
model/dropout_1/IdentityIdentity&model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/Identity?
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices?
#model/global_average_pooling2d/MeanMean!model/dropout_1/Identity:output:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2%
#model/global_average_pooling2d/Mean?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul,model/global_average_pooling2d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/BiasAdd?
&model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model/conv2d_1/Conv2D_1/ReadVariableOp?
model/conv2d_1/Conv2D_1Conv2Dinputs_1.model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D_1?
'model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model/conv2d_1/BiasAdd_1/ReadVariableOp?
model/conv2d_1/BiasAdd_1BiasAdd model/conv2d_1/Conv2D_1:output:0/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/BiasAdd_1?
model/conv2d_1/Relu_1Relu!model/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/Relu_1?
model/max_pooling2d_1/MaxPool_1MaxPool#model/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_1/MaxPool_1?
model/dropout_1/Identity_1Identity(model/max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/Identity_1?
7model/global_average_pooling2d/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/global_average_pooling2d/Mean_1/reduction_indices?
%model/global_average_pooling2d/Mean_1Mean#model/dropout_1/Identity_1:output:0@model/global_average_pooling2d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2'
%model/global_average_pooling2d/Mean_1?
#model/dense/MatMul_1/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02%
#model/dense/MatMul_1/ReadVariableOp?
model/dense/MatMul_1MatMul.model/global_average_pooling2d/Mean_1:output:0+model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/MatMul_1?
$model/dense/BiasAdd_1/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$model/dense/BiasAdd_1/ReadVariableOp?
model/dense/BiasAdd_1BiasAddmodel/dense/MatMul_1:product:0,model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/BiasAdd_1?

lambda/subSubmodel/dense/BiasAdd:output:0model/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????02

lambda/subj
lambda/SquareSquarelambda/sub:z:0*
T0*'
_output_shapes
:?????????02
lambda/Square~
lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/Sum/reduction_indices?

lambda/SumSumlambda/Square:y:0%lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

lambda/Sumi
lambda/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
lambda/Maximum/y?
lambda/MaximumMaximumlambda/Sum:output:0lambda/Maximum/y:output:0*
T0*'
_output_shapes
:?????????2
lambda/Maximuma
lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/Const?
lambda/Maximum_1Maximumlambda/Maximum:z:0lambda/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda/Maximum_1j
lambda/SqrtSqrtlambda/Maximum_1:z:0*
T0*'
_output_shapes
:?????????2
lambda/Sqrt?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullambda/Sqrt:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoidn
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp(^model/conv2d_1/BiasAdd_1/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_1/Conv2D_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/BiasAdd_1/ReadVariableOp"^model/dense/MatMul/ReadVariableOp$^model/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2R
'model/conv2d_1/BiasAdd_1/ReadVariableOp'model/conv2d_1/BiasAdd_1/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_1/Conv2D_1/ReadVariableOp&model/conv2d_1/Conv2D_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/BiasAdd_1/ReadVariableOp$model/dense/BiasAdd_1/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2J
#model/dense/MatMul_1/ReadVariableOp#model/dense/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
%__inference_model_layer_call_fn_45286
input_3!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_452622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_45533
input_1
input_2%
model_45512:@
model_45514:@
model_45516:@0
model_45518:0
dense_1_45527:
dense_1_45529:
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_45512model_45514model_45516model_45518*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_451662
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_45512model_45514model_45516model_45518*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_451662!
model/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_453572
lambda/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_1_45527dense_1_45529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_453702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
K
/__inference_max_pooling2d_1_layer_call_fn_45906

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_450682
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_45933

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_451402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_45871

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_453702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_45891

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_451232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_45377

inputs
inputs_1%
model_45329:@
model_45331:@
model_45333:@0
model_45335:0
dense_1_45371:
dense_1_45373:
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_45329model_45331model_45333model_45335*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_451662
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_45329model_45331model_45333model_45335*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_451662!
model/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_453572
lambda/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_1_45371dense_1_45373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_453702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_45508
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_454752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?_
?
B__inference_model_1_layer_call_and_return_conditional_losses_45700
inputs_0
inputs_1G
-model_conv2d_1_conv2d_readvariableop_resource:@<
.model_conv2d_1_biasadd_readvariableop_resource:@<
*model_dense_matmul_readvariableop_resource:@09
+model_dense_biasadd_readvariableop_resource:08
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?'model/conv2d_1/BiasAdd_1/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?&model/conv2d_1/Conv2D_1/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?$model/dense/BiasAdd_1/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?#model/dense/MatMul_1/ReadVariableOp?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dinputs_0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/BiasAdd?
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model/dropout_1/dropout/Const?
model/dropout_1/dropout/MulMul&model/max_pooling2d_1/MaxPool:output:0&model/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/dropout/Mul?
model/dropout_1/dropout/ShapeShape&model/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
model/dropout_1/dropout/Shape?
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype026
4model/dropout_1/dropout/random_uniform/RandomUniform?
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2(
&model/dropout_1/dropout/GreaterEqual/y?
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2&
$model/dropout_1/dropout/GreaterEqual?
model/dropout_1/dropout/CastCast(model/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
model/dropout_1/dropout/Cast?
model/dropout_1/dropout/Mul_1Mulmodel/dropout_1/dropout/Mul:z:0 model/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/dropout/Mul_1?
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices?
#model/global_average_pooling2d/MeanMean!model/dropout_1/dropout/Mul_1:z:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2%
#model/global_average_pooling2d/Mean?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul,model/global_average_pooling2d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/BiasAdd?
&model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model/conv2d_1/Conv2D_1/ReadVariableOp?
model/conv2d_1/Conv2D_1Conv2Dinputs_1.model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D_1?
'model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model/conv2d_1/BiasAdd_1/ReadVariableOp?
model/conv2d_1/BiasAdd_1BiasAdd model/conv2d_1/Conv2D_1:output:0/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/BiasAdd_1?
model/conv2d_1/Relu_1Relu!model/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/Relu_1?
model/max_pooling2d_1/MaxPool_1MaxPool#model/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_1/MaxPool_1?
model/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model/dropout_1/dropout_1/Const?
model/dropout_1/dropout_1/MulMul(model/max_pooling2d_1/MaxPool_1:output:0(model/dropout_1/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/dropout_1/Mul?
model/dropout_1/dropout_1/ShapeShape(model/max_pooling2d_1/MaxPool_1:output:0*
T0*
_output_shapes
:2!
model/dropout_1/dropout_1/Shape?
6model/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform(model/dropout_1/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype028
6model/dropout_1/dropout_1/random_uniform/RandomUniform?
(model/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model/dropout_1/dropout_1/GreaterEqual/y?
&model/dropout_1/dropout_1/GreaterEqualGreaterEqual?model/dropout_1/dropout_1/random_uniform/RandomUniform:output:01model/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2(
&model/dropout_1/dropout_1/GreaterEqual?
model/dropout_1/dropout_1/CastCast*model/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2 
model/dropout_1/dropout_1/Cast?
model/dropout_1/dropout_1/Mul_1Mul!model/dropout_1/dropout_1/Mul:z:0"model/dropout_1/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????@2!
model/dropout_1/dropout_1/Mul_1?
7model/global_average_pooling2d/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/global_average_pooling2d/Mean_1/reduction_indices?
%model/global_average_pooling2d/Mean_1Mean#model/dropout_1/dropout_1/Mul_1:z:0@model/global_average_pooling2d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2'
%model/global_average_pooling2d/Mean_1?
#model/dense/MatMul_1/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02%
#model/dense/MatMul_1/ReadVariableOp?
model/dense/MatMul_1MatMul.model/global_average_pooling2d/Mean_1:output:0+model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/MatMul_1?
$model/dense/BiasAdd_1/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$model/dense/BiasAdd_1/ReadVariableOp?
model/dense/BiasAdd_1BiasAddmodel/dense/MatMul_1:product:0,model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/BiasAdd_1?

lambda/subSubmodel/dense/BiasAdd:output:0model/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????02

lambda/subj
lambda/SquareSquarelambda/sub:z:0*
T0*'
_output_shapes
:?????????02
lambda/Square~
lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/Sum/reduction_indices?

lambda/SumSumlambda/Square:y:0%lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

lambda/Sumi
lambda/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
lambda/Maximum/y?
lambda/MaximumMaximumlambda/Sum:output:0lambda/Maximum/y:output:0*
T0*'
_output_shapes
:?????????2
lambda/Maximuma
lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/Const?
lambda/Maximum_1Maximumlambda/Maximum:z:0lambda/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda/Maximum_1j
lambda/SqrtSqrtlambda/Maximum_1:z:0*
T0*'
_output_shapes
:?????????2
lambda/Sqrt?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullambda/Sqrt:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoidn
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp(^model/conv2d_1/BiasAdd_1/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_1/Conv2D_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/BiasAdd_1/ReadVariableOp"^model/dense/MatMul/ReadVariableOp$^model/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2R
'model/conv2d_1/BiasAdd_1/ReadVariableOp'model/conv2d_1/BiasAdd_1/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_1/Conv2D_1/ReadVariableOp&model/conv2d_1/Conv2D_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/BiasAdd_1/ReadVariableOp$model/dense/BiasAdd_1/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2J
#model/dense/MatMul_1/ReadVariableOp#model/dense/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
%__inference_model_layer_call_fn_45811

inputs!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_452622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45901

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_45882

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_45140

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
T
8__inference_global_average_pooling2d_layer_call_fn_45955

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_450912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_45370

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
@__inference_model_layer_call_and_return_conditional_losses_45785

inputsA
'conv2d_1_conv2d_readvariableop_resource:@6
(conv2d_1_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@03
%dense_biasadd_readvariableop_resource:0
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/Mul_1?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeandropout_1/dropout/Mul_1:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
global_average_pooling2d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/BiasAddq
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_45979

inputs
unknown:@0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_451592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45133

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_45177
input_3!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_451662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
k
A__inference_lambda_layer_call_and_return_conditional_losses_45357

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????02
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????02
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_45262

inputs(
conv2d_1_45248:@
conv2d_1_45250:@
dense_45256:@0
dense_45258:0
identity?? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_45248conv2d_1_45250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_451232"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_451332!
max_pooling2d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_452122#
!dropout_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_451472*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_45256dense_45258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_451592
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity?
NoOpNoOp!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45068

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45896

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45944

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45147

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45950

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?<
?

__inference__traced_save_46084
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : :@:@:@0:0: : : : :::@:@:@0:0:::@:@:@0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:$
 

_output_shapes

:@0: 

_output_shapes
:0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@0: 

_output_shapes
:0:$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@0: 

_output_shapes
:0:

_output_shapes
: 
?

?
#__inference_signature_wrapper_45584
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_450592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_45475

inputs
inputs_1%
model_45454:@
model_45456:@
model_45458:@0
model_45460:0
dense_1_45469:
dense_1_45471:
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_45454model_45456model_45458model_45460*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_452622
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_45454model_45456model_45458model_45460*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_452622!
model/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_454262
lambda/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_1_45469dense_1_45471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_453702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_45212

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_45911

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_451332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_45123

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_45392
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_453772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
@__inference_model_layer_call_and_return_conditional_losses_45320
input_3(
conv2d_1_45306:@
conv2d_1_45308:@
dense_45314:@0
dense_45316:0
identity?? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_1_45306conv2d_1_45308*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_451232"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_451332!
max_pooling2d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_452122#
!dropout_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_451472*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_45314dense_45316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_451592
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity?
NoOpNoOp!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_45558
input_1
input_2%
model_45537:@
model_45539:@
model_45541:@0
model_45543:0
dense_1_45552:
dense_1_45554:
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_45537model_45539model_45541model_45543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_452622
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_45537model_45539model_45541model_45543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_452622!
model/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_454262
lambda/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_1_45552dense_1_45554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_453702!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
b
)__inference_dropout_1_layer_call_fn_45938

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_452122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
k
A__inference_lambda_layer_call_and_return_conditional_losses_45426

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????02
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????02
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_45970

inputs0
matmul_readvariableop_resource:@0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
m
A__inference_lambda_layer_call_and_return_conditional_losses_45825
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????02
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????02
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
?
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45091

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_45862

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
R
&__inference_lambda_layer_call_fn_45851
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_454262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
?

?
'__inference_model_1_layer_call_fn_45736
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@0
	unknown_2:0
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_454752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
m
A__inference_lambda_layer_call_and_return_conditional_losses_45839
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????02
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????02
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_45928

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????
C
input_28
serving_default_input_2:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
 iter

!beta_1

"beta_2
	#decay
$learning_ratemumv%mw&mx'my(mzv{v|%v}&v~'v(v?"
	optimizer
J
%0
&1
'2
(3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
%0
&1
'2
(3
4
5"
trackable_list_wrapper
?
)metrics
	variables

*layers
+layer_regularization_losses
,layer_metrics
regularization_losses
	trainable_variables
-non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?

%kernel
&bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

'kernel
(bias
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
<
%0
&1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
?
Bmetrics
	variables

Clayers
Dlayer_regularization_losses
Elayer_metrics
regularization_losses
trainable_variables
Fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gmetrics
	variables

Hlayers
Ilayer_regularization_losses
Jlayer_metrics
regularization_losses
trainable_variables
Knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Lmetrics
	variables

Mlayers
Nlayer_regularization_losses
Olayer_metrics
regularization_losses
trainable_variables
Pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'@2conv2d_1/kernel
:@2conv2d_1/bias
:@02dense/kernel
:02
dense/bias
.
Q0
R1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
Smetrics
.	variables

Tlayers
Ulayer_regularization_losses
Vlayer_metrics
/regularization_losses
0trainable_variables
Wnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xmetrics
2	variables

Ylayers
Zlayer_regularization_losses
[layer_metrics
3regularization_losses
4trainable_variables
\non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]metrics
6	variables

^layers
_layer_regularization_losses
`layer_metrics
7regularization_losses
8trainable_variables
anon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
bmetrics
:	variables

clayers
dlayer_regularization_losses
elayer_metrics
;regularization_losses
<trainable_variables
fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
gmetrics
>	variables

hlayers
ilayer_regularization_losses
jlayer_metrics
?regularization_losses
@trainable_variables
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	ltotal
	mcount
n	variables
o	keras_api"
_tf_keras_metric
^
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
l0
m1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
.:,@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
#:!@02Adam/dense/kernel/m
:02Adam/dense/bias/m
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
.:,@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
#:!@02Adam/dense/kernel/v
:02Adam/dense/bias/v
?B?
 __inference__wrapped_model_45059input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_45635
B__inference_model_1_layer_call_and_return_conditional_losses_45700
B__inference_model_1_layer_call_and_return_conditional_losses_45533
B__inference_model_1_layer_call_and_return_conditional_losses_45558?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_1_layer_call_fn_45392
'__inference_model_1_layer_call_fn_45718
'__inference_model_1_layer_call_fn_45736
'__inference_model_1_layer_call_fn_45508?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_45757
@__inference_model_layer_call_and_return_conditional_losses_45785
@__inference_model_layer_call_and_return_conditional_losses_45303
@__inference_model_layer_call_and_return_conditional_losses_45320?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_model_layer_call_fn_45177
%__inference_model_layer_call_fn_45798
%__inference_model_layer_call_fn_45811
%__inference_model_layer_call_fn_45286?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_lambda_layer_call_and_return_conditional_losses_45825
A__inference_lambda_layer_call_and_return_conditional_losses_45839?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lambda_layer_call_fn_45845
&__inference_lambda_layer_call_fn_45851?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_45862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_45871?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_45584input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_45882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_45891?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45896
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45901?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_1_layer_call_fn_45906
/__inference_max_pooling2d_1_layer_call_fn_45911?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_45916
D__inference_dropout_1_layer_call_and_return_conditional_losses_45928?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_45933
)__inference_dropout_1_layer_call_fn_45938?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45944
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_global_average_pooling2d_layer_call_fn_45955
8__inference_global_average_pooling2d_layer_call_fn_45960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_45970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_45979?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_45059?%&'(h?e
^?[
Y?V
)?&
input_1?????????
)?&
input_2?????????
? "1?.
,
dense_1!?
dense_1??????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_45882l%&7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_1_layer_call_fn_45891_%&7?4
-?*
(?%
inputs?????????
? " ??????????@?
B__inference_dense_1_layer_call_and_return_conditional_losses_45862\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_45871O/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_45970\'(/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????0
? x
%__inference_dense_layer_call_fn_45979O'(/?,
%?"
 ?
inputs?????????@
? "??????????0?
D__inference_dropout_1_layer_call_and_return_conditional_losses_45916l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_45928l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
)__inference_dropout_1_layer_call_fn_45933_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
)__inference_dropout_1_layer_call_fn_45938_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45944?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_45950`7?4
-?*
(?%
inputs?????????@
? "%?"
?
0?????????@
? ?
8__inference_global_average_pooling2d_layer_call_fn_45955wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
8__inference_global_average_pooling2d_layer_call_fn_45960S7?4
-?*
(?%
inputs?????????@
? "??????????@?
A__inference_lambda_layer_call_and_return_conditional_losses_45825?b?_
X?U
K?H
"?
inputs/0?????????0
"?
inputs/1?????????0

 
p 
? "%?"
?
0?????????
? ?
A__inference_lambda_layer_call_and_return_conditional_losses_45839?b?_
X?U
K?H
"?
inputs/0?????????0
"?
inputs/1?????????0

 
p
? "%?"
?
0?????????
? ?
&__inference_lambda_layer_call_fn_45845~b?_
X?U
K?H
"?
inputs/0?????????0
"?
inputs/1?????????0

 
p 
? "???????????
&__inference_lambda_layer_call_fn_45851~b?_
X?U
K?H
"?
inputs/0?????????0
"?
inputs/1?????????0

 
p
? "???????????
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45896?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45901h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
/__inference_max_pooling2d_1_layer_call_fn_45906?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
/__inference_max_pooling2d_1_layer_call_fn_45911[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
B__inference_model_1_layer_call_and_return_conditional_losses_45533?%&'(p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_45558?%&'(p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_45635?%&'(r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_45700?%&'(r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_1_layer_call_fn_45392?%&'(p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p 

 
? "???????????
'__inference_model_1_layer_call_fn_45508?%&'(p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p

 
? "???????????
'__inference_model_1_layer_call_fn_45718?%&'(r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "???????????
'__inference_model_1_layer_call_fn_45736?%&'(r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_45303o%&'(@?=
6?3
)?&
input_3?????????
p 

 
? "%?"
?
0?????????0
? ?
@__inference_model_layer_call_and_return_conditional_losses_45320o%&'(@?=
6?3
)?&
input_3?????????
p

 
? "%?"
?
0?????????0
? ?
@__inference_model_layer_call_and_return_conditional_losses_45757n%&'(??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????0
? ?
@__inference_model_layer_call_and_return_conditional_losses_45785n%&'(??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????0
? ?
%__inference_model_layer_call_fn_45177b%&'(@?=
6?3
)?&
input_3?????????
p 

 
? "??????????0?
%__inference_model_layer_call_fn_45286b%&'(@?=
6?3
)?&
input_3?????????
p

 
? "??????????0?
%__inference_model_layer_call_fn_45798a%&'(??<
5?2
(?%
inputs?????????
p 

 
? "??????????0?
%__inference_model_layer_call_fn_45811a%&'(??<
5?2
(?%
inputs?????????
p

 
? "??????????0?
#__inference_signature_wrapper_45584?%&'(y?v
? 
o?l
4
input_1)?&
input_1?????????
4
input_2)?&
input_2?????????"1?.
,
dense_1!?
dense_1?????????