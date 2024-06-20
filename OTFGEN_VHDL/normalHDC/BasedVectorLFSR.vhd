LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY BasedVectorLFSR IS
	GENERIC ( n	: INTEGER	:= 2000			    -- number of bits
				);	 
	PORT (
		clk, rst, update	: IN STD_LOGIC; 
		congigSignitureIn , congigInitialvaluesIn : IN STD_LOGIC_VECTOR (n-1 DOWNTO 0 );
		dout				: OUT  STD_LOGIC_VECTOR (n-1 DOWNTO 0 )
	);
END ENTITY BasedVectorLFSR;

ARCHITECTURE behavioral OF BasedVectorLFSR IS

COMPONENT  regOne IS
	GENERIC (init : STD_LOGIC := '1');   -- initial value
	PORT (
		clk 				: IN  STD_LOGIC;
		regUpdate, regrst 	: IN  STD_LOGIC;
		din        			: IN  STD_LOGIC;
		dout        		: OUT STD_LOGIC
	);
END COMPONENT;

SIGNAL interValIn : STD_LOGIC_VECTOR (n-1 DOWNTO 0);
SIGNAL interValOut : STD_LOGIC_VECTOR (n-1 DOWNTO 0);
SIGNAL interValOutR : STD_LOGIC_VECTOR (0 TO n-1);
CONSTANT congigSigniture : STD_LOGIC_VECTOR (n-1 DOWNTO 0) :=    "";

CONSTANT congigInitialvalues : STD_LOGIC_VECTOR (n-1 DOWNTO 0) := "";

BEGIN

	regArr : FOR i IN 1 TO n-1 GENERATE
		Ireg : regOne 
			GENERIC map(congigInitialvalues(i))
			PORT map(clk,  update, rst, 
			interValIn(i), 	interValOut(i) 
					 );   
	END GENERATE regArr;
	xorchain : FOR i IN 1 TO n-1 GENERATE
	xorchain1: IF (congigSigniture(i) = '1') GENERATE
			interValIn(i) <= interValOut(i-1) XOR  interValOut(n-1);
		END GENERATE xorchain1;
	xorchain2: IF (congigSigniture(i) = '0') GENERATE
			interValIn(i) <= interValOut(i-1);
		END GENERATE xorchain2;
	END GENERATE xorchain;
	interValIn(0) <= interValOut(n-1);
	reg0 : regOne 
			GENERIC map(congigInitialvalues(0))
			PORT map(clk,  update, rst, 
			interValIn(0), 	interValOut(0) 
					 ); 
	--interValOutR <= interValOut;
	dout <= interValOut;--(0 TO n-1);

END ARCHITECTURE behavioral;