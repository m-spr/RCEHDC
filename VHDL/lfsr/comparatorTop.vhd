-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY comparatorTop  IS
	GENERIC (len : INTEGER := 8;   -- bit width out adder
			 n : INTEGER := 10; 	-- #Classes       ---- all 4s in this code can be replaced with LOG2(n)
			 z : INTEGER := 10; 	-- zeropadding to 2**
			 lgn : INTEGER := 4); 	---LOG2(n)
	PORT (
		clk, rst, run  	: IN STD_LOGIC;
		a        		: IN  STD_LOGIC_VECTOR (n*len - 1 DOWNTO 0);    --- 16 = 2**4 ,,, 4 is LOG2(n)
		done  , TLAST_S, TVALID_S        	: OUT  STD_LOGIC;    							 --- final result is ready 
		classIndex 		: OUT  STD_LOGIC_VECTOR  (lgn-1 DOWNTO 0)  			 --- only the index of class can be also the value!  As of now only support up to 16 classes so 4'bits 
			);
END ENTITY comparatorTop ;

ARCHITECTURE behavioral OF comparatorTop  IS
	COMPONENT recMux IS
		GENERIC ( n : INTEGER := 3;			-- MUX sel bit width (number of layer)
				  lenPop : INTEGER := 4);   -- bit width out popCounters
		PORT (
			sel 		: IN STD_LOGIC_VECTOR (n DOWNTO 0);       -- sel<='0'&&sel
			din			: IN  STD_LOGIC_VECTOR (((2**n)*lenPop)- 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		); 
	END COMPONENT;
	COMPONENT comparator  IS
		GENERIC (len : INTEGER := 8);   -- bit width out adder
		PORT (
			a, b         : IN  STD_LOGIC_VECTOR (len - 1 DOWNTO 0);
			gr        : OUT  STD_LOGIC      --- case of a >= b gr ='1' other wise 0
		);
	END COMPONENT;
	COMPONENT confCompCtrl IS
		GENERIC (n : INTEGER := 10 ; 	-- #Classes
				lgn : INTEGER := 4 );
		PORT (
			clk, rst 				: IN STD_LOGIC;
			run		 				: IN STD_LOGIC;
			runOut, done , TLAST_S, TVALID_S 			: OUT STD_LOGIC;
			pointer 				: OUT STD_LOGIC_VECTOR(lgn-1 DOWNTO 0) --- As of now only support up to 16 classes so 4'bits 
		);
	END COMPONENT;
	COMPONENT reg IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk 	: IN STD_LOGIC;
			regUpdate, regrst 	: IN STD_LOGIC;
			din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		);
	END COMPONENT;
	COMPONENT reg1 IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk 	: IN STD_LOGIC;
			regUpdate, regrst 	: IN STD_LOGIC;
			din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		);
	END COMPONENT;
	
	SIGNAL muxSel : STD_LOGIC_VECTOR (lgn-1 DOWNTO 0);
	SIGNAL classIndexI, classIndexI2 	: STD_LOGIC_VECTOR  (lgn-1 DOWNTO 0);
	SIGNAL muxSelE : STD_LOGIC_VECTOR (lgn DOWNTO 0);
	SIGNAL muxOut, toComp, fromComp : STD_LOGIC_VECTOR (len-1 DOWNTO 0);
	SIGNAL muxIn : STD_LOGIC_VECTOR (((2**lgn)+1)*len -1 DOWNTO 0);
	CONSTANT zero_muxin : STD_LOGIC_VECTOR (((z+1)*len) -1 DOWNTO 0) := (OTHERS => '0');
	SIGNAL regIndexupd, regIndexup, regUpdate, regrst, doneI : STD_LOGIC;
	
-- attribute MARK_DEBUG : string;
-- attribute MARK_DEBUG of regUpdate : signal is "TRUE";
-- attribute MARK_DEBUG of muxSel : signal is "TRUE";
-- attribute MARK_DEBUG of toComp : signal is "TRUE";
-- attribute MARK_DEBUG of classIndexI2 : signal is "TRUE";
-- attribute MARK_DEBUG of muxOut : signal is "TRUE";
BEGIN
	muxIn <=  a & zero_muxin;   ------- check!!!
	
	ctrl : confCompCtrl 
		GENERIC MAP( n, lgn )	
		PORT MAP(
			clk, rst, run,	
			regrst, doneI, TLAST_S, TVALID_S, muxSel 
		);
	
	CompMux : recMux 
		GENERIC MAP( lgn,	
				   len)
		PORT MAP(
			muxSelE, muxIn(((2**lgn)+1)*len - 1 DOWNTO ((2**lgn)+1)*len - ((2**lgn)*len)), muxOut ------- check!!!
		);
		
	regMax : reg1 
		GENERIC MAP (len)
		PORT MAP (clk, regUpdate, regrst, muxOut, toComp);
		
	regIndexI : reg 
		GENERIC MAP (lgn)
		PORT MAP (clk, regUpdate , rst, muxSel, classIndexI);
	
	regIndex : reg 
		GENERIC MAP (lgn)
		PORT MAP (clk, doneI , rst, classIndexI2,classIndex );
		
	comp : comparator 
		GENERIC MAP(len)
		PORT MAP(
			toComp ,muxOut,
			regUpdate     --- case of a >= b gr ='1' other wise 0
		);	
	
	done <= doneI;
	muxSelE <= '0'&muxSel;
	classIndexI2 <= std_logic_vector(unsigned(n-1-unsigned(classIndexI)));
END ARCHITECTURE behavioral;
